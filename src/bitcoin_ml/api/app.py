"""Research dashboard API: latest completed BTCUSDT USD-M candle estimate."""

from __future__ import annotations

from contextlib import asynccontextmanager
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

import ccxt
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.bitcoin_ml.features.build_features import build_features
from src.bitcoin_ml.features.build_advanced_features import build_advanced_features


ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "artifacts/models/futures_random_forest_v1.joblib"
MODEL_INFO_PATH = ROOT / "artifacts/models/futures_random_forest_v1.json"
ADVANCED_MODEL_INFO_PATH = ROOT / "artifacts/models/advanced_ensemble_v1.json"
ADVANCED_TEST_METRICS_PATH = ROOT / "artifacts/metrics/advanced_final_test_model_metrics.csv"
ADVANCED_STRATEGY_PATH = ROOT / "artifacts/metrics/advanced_final_test_strategy_grid.csv"
ADVANCED_CONFIDENCE_PATH = ROOT / "artifacts/metrics/advanced_final_test_confidence.csv"
PRICE_MODEL_PATH = ROOT / "artifacts/models/futures_ohlc_regressor_v1.joblib"
PRICE_MODEL_INFO_PATH = ROOT / "artifacts/models/futures_ohlc_regressor_v1.json"
TEST_METRICS_PATH = ROOT / "artifacts/metrics/futures_final_test_model_metrics.csv"
STRATEGY_PATH = ROOT / "artifacts/metrics/futures_final_test_strategy_grid.csv"
PRICE_REGRESSION_METRICS_PATH = ROOT / "artifacts/metrics/futures_ohlc_regression_metrics.json"
MONTHLY_PATH = ROOT / "artifacts/metrics/futures_monitoring_monthly.csv"
PREDICTIONS_DB = ROOT / "artifacts/registry/live_predictions.sqlite3"
TIMEFRAME = "1h"
MAX_DATA_AGE = timedelta(minutes=75)


class PredictionResponse(BaseModel):
    status: str
    symbol: str
    timeframe: str
    direction: str | None
    probability_up: float | None
    probability_down: float | None
    model_version: str
    last_closed_candle_open_utc: datetime | None
    last_closed_candle_close_utc: datetime | None
    predicted_candle_open_utc: datetime | None
    predicted_candle_close_utc: datetime | None
    explanation: str
    current_price: float | None
    price_change_24h_pct: float | None
    price_as_of_utc: datetime | None
    price_source: str
    data_as_of_utc: datetime
    data_age_minutes: float | None
    freshness: str
    warning: str
    estimated_open_price: float | None = None
    estimated_close_price: float | None = None
    estimated_high_price: float | None = None
    estimated_low_price: float | None = None
    estimated_return_pct: float | None = None
    price_estimate_method: str | None = None
    price_model_version: str | None = None
    ensemble_probabilities: dict[str, float] | None = None
    confidence_level: str | None = None


class DashboardResponse(BaseModel):
    prediction: PredictionResponse
    recent_predictions: list[dict[str, Any]]
    final_test: dict[str, Any]
    strategy_scenarios: list[dict[str, Any]]
    monitoring: dict[str, Any]
    price_model_test_metrics: dict[str, Any]
    confidence_analysis: list[dict[str, Any]] | None = None
    system: dict[str, Any]


class MarketCandle(BaseModel):
    # `time` is Unix seconds, which candlestick chart libraries can consume directly.
    time: int
    timestamp_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool


class MarketCandlesResponse(BaseModel):
    symbol: str
    timeframe: str
    as_of_utc: datetime
    candles: list[MarketCandle]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check for advanced multi-model ensemble first
    app.state.is_advanced = ADVANCED_MODEL_INFO_PATH.exists()
    if app.state.is_advanced:
        app.state.advanced_info = json.loads(ADVANCED_MODEL_INFO_PATH.read_text(encoding="utf-8"))
        app.state.advanced_models = {}
        for m_name, m_file in app.state.advanced_info.get("models", {}).items():
            m_path = ROOT / "artifacts/models" / m_file
            if m_path.exists():
                app.state.advanced_models[m_name] = joblib.load(m_path)
        medians_path = ROOT / "artifacts/models" / app.state.advanced_info.get("feature_medians_path", "")
        app.state.feature_medians = joblib.load(medians_path) if medians_path.exists() else {}
        if ADVANCED_TEST_METRICS_PATH.exists():
            metrics_df = pd.read_csv(ADVANCED_TEST_METRICS_PATH)
            weights = {
                row["model"]: float(row["roc_auc"])
                for _, row in metrics_df.iterrows()
                if row["model"] != "ensemble" and row["model"] in app.state.advanced_models
            }
            total_w = sum(weights.values()) or 1.0
            app.state.ensemble_weights = {k: v / total_w for k, v in weights.items()}
        else:
            app.state.ensemble_weights = {k: 1.0 / len(app.state.advanced_models) for k in app.state.advanced_models}
        app.state.model_info = app.state.advanced_info
    elif MODEL_PATH.exists() and MODEL_INFO_PATH.exists():
        app.state.model = joblib.load(MODEL_PATH)
        app.state.model_info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
    else:
        raise RuntimeError(
            f"Saved model not found. Train either the baseline or advanced ensemble first: {ADVANCED_MODEL_INFO_PATH}"
        )

    if not PRICE_MODEL_PATH.exists() or not PRICE_MODEL_INFO_PATH.exists():
        raise RuntimeError(
            f"OHLC price model not found. Train it first: {PRICE_MODEL_PATH}"
        )
    app.state.price_model = joblib.load(PRICE_MODEL_PATH)
    app.state.price_model_info = json.loads(PRICE_MODEL_INFO_PATH.read_text(encoding="utf-8"))
    app.state.exchange = ccxt.binanceusdm({"enableRateLimit": True})
    yield
    app.state.exchange.close()


app = FastAPI(
    title="Bitcoin ML Intelligence API",
    version="1.0.0",
    description="Research estimates from a saved BTC/USDT USD-M Futures model.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def open_prediction_db() -> sqlite3.Connection:
    PREDICTIONS_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(PREDICTIONS_DB, timeout=10)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS live_predictions (
            prediction_candle_open_utc TEXT PRIMARY KEY,
            prediction_candle_close_utc TEXT NOT NULL,
            input_candle_open_utc TEXT NOT NULL,
            input_candle_close_utc TEXT NOT NULL,
            input_close REAL NOT NULL,
            probability_up REAL NOT NULL,
            direction TEXT NOT NULL,
            confidence_level TEXT,
            model_version TEXT NOT NULL,
            generated_at_utc TEXT NOT NULL,
            actual_up INTEGER,
            actual_return REAL,
            predicted_open REAL,
            predicted_high REAL,
            predicted_low REAL,
            predicted_close REAL,
            predicted_return_pct REAL,
            actual_open REAL,
            actual_high REAL,
            actual_low REAL,
            actual_close REAL,
            outcome_recorded_at_utc TEXT
        )
    """)
    # Upgrade existing local prediction databases without deleting their history.
    existing = {row[1] for row in connection.execute("PRAGMA table_info(live_predictions)")}
    columns_to_add = [
        ("predicted_open", "REAL"),
        ("predicted_high", "REAL"),
        ("predicted_low", "REAL"),
        ("predicted_close", "REAL"),
        ("predicted_return_pct", "REAL"),
        ("actual_open", "REAL"),
        ("actual_high", "REAL"),
        ("actual_low", "REAL"),
        ("actual_close", "REAL"),
        ("confidence_level", "TEXT"),
    ]
    for column, col_type in columns_to_add:
        if column not in existing:
            connection.execute(f"ALTER TABLE live_predictions ADD COLUMN {column} {col_type}")
    return connection


def record_prediction(prediction: PredictionResponse, input_close: float) -> None:
    if prediction.status != "available":
        return
    with closing(open_prediction_db()) as db, db:
        db.execute("""
            INSERT INTO live_predictions (
                prediction_candle_open_utc, prediction_candle_close_utc,
                input_candle_open_utc, input_candle_close_utc, input_close,
                probability_up, direction, confidence_level, model_version, generated_at_utc,
                predicted_open, predicted_high, predicted_low, predicted_close, predicted_return_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(prediction_candle_open_utc) DO UPDATE SET
                predicted_open=excluded.predicted_open,
                predicted_high=excluded.predicted_high,
                predicted_low=excluded.predicted_low,
                predicted_close=excluded.predicted_close,
                predicted_return_pct=excluded.predicted_return_pct,
                confidence_level=excluded.confidence_level,
                probability_up=excluded.probability_up,
                direction=excluded.direction,
                model_version=excluded.model_version
            WHERE actual_up IS NULL
        """, (
            prediction.predicted_candle_open_utc.isoformat(),
            prediction.predicted_candle_close_utc.isoformat(),
            prediction.last_closed_candle_open_utc.isoformat(),
            prediction.last_closed_candle_close_utc.isoformat(), input_close,
            prediction.probability_up, prediction.direction, prediction.confidence_level, prediction.model_version,
            prediction.data_as_of_utc.isoformat(),
            prediction.estimated_open_price, prediction.estimated_high_price,
            prediction.estimated_low_price, prediction.estimated_close_price,
            prediction.estimated_return_pct,
        ))


def settle_completed_predictions(candles: pd.DataFrame) -> None:
    """Settle stored estimates only against the matching, completed candle."""
    candles_by_open = {
        timestamp.to_pydatetime().isoformat(): (float(open_), float(high), float(low), float(close))
        for timestamp, open_, high, low, close in zip(
            candles["timestamp"], candles["open"], candles["high"], candles["low"], candles["close"]
        )
    }
    with closing(open_prediction_db()) as db, db:
        pending = db.execute(
            "SELECT prediction_candle_open_utc, input_close FROM live_predictions WHERE actual_up IS NULL"
        ).fetchall()
        for key, input_close in pending:
            actual_candle = candles_by_open.get(key)
            if actual_candle is None:
                continue
            open_price, high_price, low_price, close_price = actual_candle
            outcome_return = close_price / input_close - 1
            db.execute("""
                UPDATE live_predictions
                SET actual_up=?, actual_return=?, actual_open=?, actual_high=?, actual_low=?, actual_close=?,
                    outcome_recorded_at_utc=?
                WHERE prediction_candle_open_utc=? AND actual_up IS NULL
            """, (int(outcome_return > 0), outcome_return, open_price, high_price, low_price,
                  close_price, utc_now().isoformat(), key))


def recent_prediction_rows(limit: int = 15) -> list[dict[str, Any]]:
    if not PREDICTIONS_DB.exists():
        return []
    with closing(sqlite3.connect(PREDICTIONS_DB)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("""
            SELECT prediction_candle_open_utc, prediction_candle_close_utc,
                   direction, probability_up, confidence_level, actual_up, actual_return,
                   model_version, generated_at_utc,
                   predicted_open, predicted_high, predicted_low, predicted_close, predicted_return_pct,
                   actual_open, actual_high, actual_low, actual_close
            FROM live_predictions
            ORDER BY prediction_candle_open_utc DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(row) for row in rows]
    return [dict(row) for row in rows]


def read_first_csv_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "note": f"Research file not found: {path.name}"}
    row = pd.read_csv(path).iloc[0].to_dict()
    return {key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
            for key, value in row.items()}


def get_prediction(request: Request) -> PredictionResponse:
    now = utc_now()
    info = request.app.state.model_info
    candle_limit = 120 if getattr(request.app.state, "is_advanced", False) else 80
    try:
        raw = request.app.state.exchange.fetch_ohlcv("BTC/USDT:USDT", TIMEFRAME, limit=candle_limit)
    except Exception as exc:  # exchange/network state is represented explicitly to the UI
        return PredictionResponse(
            status="unavailable", symbol="BTC/USDT USD-M", timeframe=TIMEFRAME,
            direction=None, probability_up=None, probability_down=None,
            model_version=info["model_version"], last_closed_candle_open_utc=None,
            last_closed_candle_close_utc=None, predicted_candle_open_utc=None,
            predicted_candle_close_utc=None, explanation="Live candle request failed.",
            current_price=None, price_change_24h_pct=None, price_as_of_utc=None,
            price_source="unavailable", data_as_of_utc=now,
            data_age_minutes=None, freshness="unavailable",
            warning=f"Exchange data unavailable: {type(exc).__name__}. Retry shortly.",
        )

    candles = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], unit="ms", utc=True)
    current_hour_open = pd.Timestamp(now).floor("h")
    # Exchange timestamps identify candle OPEN. Exclude the still-forming hour.
    candles = candles.loc[candles["timestamp"] < current_hour_open].copy()
    if candles.empty:
        raise HTTPException(status_code=503, detail="No completed hourly candle is available yet.")
    candles = candles.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    settle_completed_predictions(candles)
    latest = candles.iloc[-1]
    candle_open = latest["timestamp"].to_pydatetime()
    candle_close = candle_open + timedelta(hours=1)
    prediction_open = candle_close
    prediction_close = prediction_open + timedelta(hours=1)
    age_minutes = max(0.0, (now - candle_close).total_seconds() / 60)
    fresh = age_minutes <= MAX_DATA_AGE.total_seconds() / 60
    has_hourly_gap = len(candles) > 1 and candles["timestamp"].diff().iloc[-1] != pd.Timedelta(hours=1)

    is_adv = getattr(request.app.state, "is_advanced", False)
    if is_adv:
        adv_features = build_advanced_features(candles)
        feature_names = info["features"]
        medians = getattr(request.app.state, "feature_medians", {})
        row_features = adv_features[feature_names].tail(1).fillna(medians)
        missing = [name for name in feature_names if name not in adv_features.columns or pd.isna(row_features.iloc[0][name])]
    else:
        basic_features = build_features(candles)
        feature_names = info["features"]
        row_features = basic_features[feature_names].tail(1)
        missing = [name for name in feature_names if name not in row_features.columns or pd.isna(row_features.iloc[0][name])]

    input_close = float(latest["close"])
    current_price = input_close
    last_24 = candles.iloc[-25] if len(candles) >= 25 else None
    change_24h = (input_close / float(last_24["close"]) - 1) * 100 if last_24 is not None else None
    price_as_of = candle_close
    price_source = "last_completed_candle"
    try:
        ticker = request.app.state.exchange.fetch_ticker("BTC/USDT:USDT")
        if ticker.get("last") is not None and float(ticker["last"]) > 0:
            current_price = float(ticker["last"])
            price_source = "live_ticker"
            price_as_of = (datetime.fromtimestamp(ticker["timestamp"] / 1000, tz=timezone.utc)
                           if ticker.get("timestamp") else now)
            if ticker.get("percentage") is not None:
                change_24h = float(ticker["percentage"])
    except Exception:
        # Keep the completed-candle close visible and identify it as the fallback source.
        pass
    if not fresh or has_hourly_gap or missing:
        reason = "stale market data" if not fresh else "latest hourly candle follows a data gap" if has_hourly_gap else "not enough consecutive candles for model features"
        return PredictionResponse(
            status="stale" if not fresh or has_hourly_gap else "warming_up",
            symbol="BTC/USDT USD-M", timeframe=TIMEFRAME, direction=None,
            probability_up=None, probability_down=None, model_version=info["model_version"],
            last_closed_candle_open_utc=candle_open, last_closed_candle_close_utc=candle_close,
            predicted_candle_open_utc=prediction_open, predicted_candle_close_utc=prediction_close,
            explanation=f"Latest candle is {candle_open:%Y-%m-%d %H:%M}–{candle_close:%H:%M} UTC; model estimates the following candle {prediction_open:%H:%M}–{prediction_close:%H:%M} UTC.",
            current_price=current_price, price_change_24h_pct=change_24h,
            price_as_of_utc=price_as_of, price_source=price_source, data_as_of_utc=now,
            data_age_minutes=round(age_minutes, 1), freshness="fresh" if fresh else "stale",
            warning=f"Estimate withheld because of {reason}. Missing model inputs: {', '.join(missing) or 'none'}.",
        )

    ensemble_probs: dict[str, float] = {}
    if is_adv:
        models = request.app.state.advanced_models
        weights = getattr(request.app.state, "ensemble_weights", {})
        total_w = sum(weights.get(m, 1.0) for m in models) or 1.0
        p_up = 0.0
        for m_name, model in models.items():
            prob = float(model.predict_proba(row_features)[0][1])
            ensemble_probs[m_name] = round(prob, 4)
            w = weights.get(m_name, 1.0)
            p_up += (w / total_w) * prob
        p_down = 1.0 - p_up
        direction = "UP" if p_up >= 0.5 else "DOWN"
        margin = abs(p_up - 0.5)
        conf_tier = (
            "High (>=65% threshold)" if margin >= 0.15
            else "Moderate (>=58% threshold)" if margin >= 0.08
            else "Standard (>=52% threshold)" if margin >= 0.02
            else "Low/Neutral (~50%)"
        )
    else:
        probabilities = request.app.state.model.predict_proba(row_features)[0]
        classes = list(request.app.state.model.classes_)
        p_up = float(probabilities[classes.index(1)])
        p_down = 1.0 - p_up
        direction = "UP" if p_up >= 0.5 else "DOWN"
        conf_tier = "Standard"

    # Directional and physical coherence between classifier and price forecast:
    features_basic = build_features(candles)
    price_info = request.app.state.price_model_info
    price_features = price_info["features"]
    predicted_ohlc_returns = request.app.state.price_model.predict(
        features_basic[price_features].tail(1)
    )[0]

    # Target candle starts at the last completed candle close
    estimated_open = input_close

    # Measure recent hourly volatility (ATR) for realistic price bounds
    if len(candles) >= 14:
        recent_ranges = candles["high"].tail(14).sub(candles["low"].tail(14))
        atr_pct = float(recent_ranges.mean() / input_close)
    else:
        atr_pct = 0.005

    # Scale expected move by ensemble confidence (higher confidence = more decisive move)
    confidence_scale = max(0.3, min(1.5, abs(p_up - 0.5) * 4.0))
    raw_return_mag = abs(float(predicted_ohlc_returns[3]))
    expected_move = max(raw_return_mag, atr_pct * 0.45) * confidence_scale

    if direction == "DOWN":
        estimated_return = -expected_move
        estimated_close = input_close * (1.0 + estimated_return)
        estimated_high = input_close * (1.0 + expected_move * 0.25)
        estimated_low = estimated_close * (1.0 - expected_move * 0.25)
    else:
        estimated_return = expected_move
        estimated_close = input_close * (1.0 + estimated_return)
        estimated_low = input_close * (1.0 - expected_move * 0.25)
        estimated_high = estimated_close * (1.0 + expected_move * 0.25)

    estimated_high = max(estimated_high, estimated_open, estimated_close)
    estimated_low = min(estimated_low, estimated_open, estimated_close)

    result = PredictionResponse(
        status="available", symbol="BTC/USDT USD-M", timeframe=TIMEFRAME,
        direction=direction, probability_up=p_up, probability_down=p_down,
        model_version=info["model_version"], last_closed_candle_open_utc=candle_open,
        last_closed_candle_close_utc=candle_close, predicted_candle_open_utc=prediction_open,
        predicted_candle_close_utc=prediction_close,
        explanation=(f"Estimate generated after candle {candle_open:%Y-%m-%d %H:%M}–{candle_close:%H:%M} UTC closed; "
                     f"it applies to candle {prediction_open:%Y-%m-%d %H:%M}–{prediction_close:%H:%M} UTC."),
        current_price=current_price, price_change_24h_pct=change_24h,
        price_as_of_utc=price_as_of, price_source=price_source,
        data_as_of_utc=now, data_age_minutes=round(age_minutes, 1), freshness="fresh",
        warning="Research estimate only. It is not an order instruction or a promise of profit.",
        estimated_open_price=round(estimated_open, 2),
        estimated_close_price=round(estimated_close, 2),
        estimated_high_price=round(estimated_high, 2),
        estimated_low_price=round(estimated_low, 2),
        estimated_return_pct=round(estimated_return * 100, 3),
        price_estimate_method="Ensemble-direction aligned multi-output return estimate with ATR volatility bounds.",
        price_model_version=price_info["model_version"],
        ensemble_probabilities=ensemble_probs if is_adv else None,
        confidence_level=conf_tier if is_adv else None,
    )
    record_prediction(result, input_close)
    return result


@app.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    return {"status": "online", "model_version": request.app.state.model_info["model_version"],
            "time_utc": utc_now().isoformat()}


@app.get("/api/prediction/latest", response_model=PredictionResponse)
def latest_prediction(request: Request) -> PredictionResponse:
    return get_prediction(request)


@app.get("/api/market/candles", response_model=MarketCandlesResponse)
def market_candles(
    request: Request,
    limit: int = Query(default=168, ge=24, le=500),
) -> MarketCandlesResponse:
    """Return recent Binance USD-M hourly candles, including the forming candle."""
    now = utc_now()
    try:
        raw = request.app.state.exchange.fetch_ohlcv(
            "BTC/USDT:USDT", TIMEFRAME, limit=limit
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Binance candle data unavailable ({type(exc).__name__}); retry shortly.",
        ) from exc

    candles: list[MarketCandle] = []
    current_hour_open = pd.Timestamp(now).floor("h")
    for timestamp_ms, open_, high, low, close, volume in raw:
        candle_open = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        candles.append(MarketCandle(
            time=int(timestamp_ms // 1000),
            timestamp_utc=candle_open,
            open=float(open_), high=float(high), low=float(low),
            close=float(close), volume=float(volume),
            is_closed=candle_open + timedelta(hours=1) <= current_hour_open,
        ))

    if not candles:
        raise HTTPException(status_code=503, detail="Binance returned no hourly candles.")
    return MarketCandlesResponse(
        symbol="BTC/USDT USD-M", timeframe=TIMEFRAME,
        as_of_utc=now, candles=candles,
    )


@app.get("/api/dashboard", response_model=DashboardResponse)
def dashboard(request: Request) -> DashboardResponse:
    prediction = get_prediction(request)
    is_adv = getattr(request.app.state, "is_advanced", False)

    if is_adv and ADVANCED_TEST_METRICS_PATH.exists():
        metrics_df = pd.read_csv(ADVANCED_TEST_METRICS_PATH)
        ensemble_row = metrics_df.loc[metrics_df["model"] == "ensemble"]
        if not ensemble_row.empty:
            final_test = ensemble_row.iloc[0].to_dict()
            final_test = {
                key: (None if pd.isna(val) else val.item() if hasattr(val, "item") else val)
                for key, val in final_test.items()
            }
        else:
            final_test = read_first_csv_row(TEST_METRICS_PATH)
    else:
        final_test = read_first_csv_row(TEST_METRICS_PATH)

    strategy_path = ADVANCED_STRATEGY_PATH if is_adv and ADVANCED_STRATEGY_PATH.exists() else STRATEGY_PATH
    strategy = pd.read_csv(strategy_path) if strategy_path.exists() else pd.DataFrame()
    if not strategy.empty:
        if "cost_scenario" in strategy.columns:
            strategy = strategy.loc[strategy["cost_scenario"] == "fee_0.0010_slippage_0.0005"]
        elif "fee_per_side_pct" in strategy.columns:
            strategy = strategy.loc[(strategy["fee_per_side_pct"] == 0.1) & (strategy["slippage_per_side_pct"] == 0.05)]
    strategy_rows = strategy[["threshold", "time_in_market_pct", "long_hours", "short_hours", "total_return_pct", "max_drawdown_pct"]].to_dict("records") if not strategy.empty else []

    confidence_rows = []
    if is_adv and ADVANCED_CONFIDENCE_PATH.exists():
        conf_df = pd.read_csv(ADVANCED_CONFIDENCE_PATH)
        confidence_rows = conf_df.to_dict("records")

    monitoring = {"available": False}
    if MONTHLY_PATH.exists():
        monthly = pd.read_csv(MONTHLY_PATH)
        monitoring = {"available": True, "months_reported": len(monthly),
                      "latest": monthly.tail(1).replace({float("nan"): None}).to_dict("records")[0]}
    price_model_test_metrics: dict[str, Any] = {"available": False}
    if PRICE_REGRESSION_METRICS_PATH.exists():
        regression = json.loads(PRICE_REGRESSION_METRICS_PATH.read_text(encoding="utf-8"))
        selected = regression["selected_model"]
        close_scores = {
            row["model"]: row
            for row in regression["test_metrics"]
            if row["target"] == "next_close_return"
        }
        price_model_test_metrics = {
            "available": selected in close_scores,
            "selected_model": selected,
            "test_rows": regression["test_rows"],
            "test_period": regression["test_period"],
            "selected_close_mae_bps": close_scores.get(selected, {}).get("mae_basis_points"),
            "selected_close_mae_dollars_at_80000": close_scores.get(selected, {}).get("mae_dollars_at_80000"),
            "baseline_close_mae_bps": close_scores.get("persistence_baseline", {}).get("mae_basis_points"),
            "baseline_close_mae_dollars_at_80000": close_scores.get("persistence_baseline", {}).get("mae_dollars_at_80000"),
            "selection_metric": regression["selection_metric"],
        }
    sys_info = request.app.state.model_info
    return DashboardResponse(
        prediction=prediction, recent_predictions=recent_prediction_rows(),
        final_test=final_test, strategy_scenarios=strategy_rows,
        monitoring=monitoring,
        price_model_test_metrics=price_model_test_metrics,
        confidence_analysis=confidence_rows if confidence_rows else None,
        system={"model_version": sys_info["model_version"],
                "trained_at_utc": sys_info.get("trained_at_utc"),
                "training_rows": sys_info.get("training_rows"),
                "training_start": sys_info.get("training_start"),
                "training_end": sys_info.get("training_end"),
                "last_refresh_utc": utc_now().isoformat(),
                "final_holdout_note": sys_info.get("final_holdout_note", "One-time holdout evaluated on reserved 15% period."),
                "usage": sys_info.get("usage", "Research estimate only."),
                "models": list(request.app.state.advanced_models.keys()) if is_adv else ["random_forest"]},
    )
