"""Train and evaluate a separate next-hour OHLC return regressor.

This model estimates a candle's open/high/low/close relative to the last
completed close. It is evaluated against a no-change baseline on a later
chronological holdout before the production artifact is refit on all rows.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor


DATA_PATH = Path("data/processed/btcusdt_usdm_1h_features.csv")
MODEL_PATH = Path("artifacts/models/futures_ohlc_regressor_v1.joblib")
METADATA_PATH = Path("artifacts/models/futures_ohlc_regressor_v1.json")
METRICS_PATH = Path("artifacts/metrics/futures_ohlc_regression_metrics.json")
PREDICTIONS_PATH = Path("artifacts/metrics/futures_ohlc_regression_test_predictions.csv")
FEATURES = [
    "return_1h", "return_3h", "return_6h", "return_24h", "volatility_24h",
    "volume_log_change_1h", "candle_body_pct", "candle_range_pct",
    "upper_wick_pct", "lower_wick_pct",
]
TARGETS = ["next_open_return", "next_high_return", "next_low_return", "next_close_return"]
TARGET_PRICES = ["actual_next_open", "actual_next_high", "actual_next_low", "actual_next_close"]
RANDOM_FOREST_PARAMETERS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 30,
    "max_features": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_model(kind: str):
    if kind == "random_forest_regressor":
        return RandomForestRegressor(**RANDOM_FOREST_PARAMETERS)
    return MultiOutputRegressor(DummyRegressor(strategy="constant", constant=0.0))


def add_next_ohlc_targets(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy().sort_values("timestamp").reset_index(drop=True)
    next_timestamp = result["timestamp"].shift(-1)
    consecutive = next_timestamp.sub(result["timestamp"]).eq(pd.Timedelta(hours=1))
    reference_close = result["close"]
    for source, target, price_target in zip(
        ["open", "high", "low", "close"], TARGETS, TARGET_PRICES
    ):
        next_price = result[source].shift(-1).where(consecutive)
        result[price_target] = next_price
        result[target] = next_price.div(reference_close).sub(1)
    return result


def metric_rows(kind: str, actual: np.ndarray, predicted: np.ndarray) -> list[dict]:
    rows = []
    for index, (target, price_name) in enumerate(zip(TARGETS, TARGET_PRICES)):
        error_bps = mean_absolute_error(actual[:, index], predicted[:, index]) * 10_000
        rows.append({
            "model": kind,
            "target": target,
            "mae_basis_points": float(error_bps),
            "mae_dollars_at_80000": float(error_bps / 10_000 * 80_000),
            "observations": int(len(actual)),
        })
    return rows


def constrain_ohlc_returns(predicted_returns: np.ndarray, reference_close: np.ndarray) -> np.ndarray:
    """Enforce high >= open/close and low <= open/close in return space."""
    prices = reference_close[:, None] * (1.0 + predicted_returns)
    candle_high = prices.max(axis=1)
    candle_low = prices.min(axis=1)
    prices[:, 1] = candle_high
    prices[:, 2] = candle_low
    return prices / reference_close[:, None] - 1.0


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Run the Futures feature pipeline first: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = add_next_ohlc_targets(data)
    data = data.dropna(subset=FEATURES + TARGETS).reset_index(drop=True)
    if len(data) < 1000:
        raise ValueError(f"Not enough valid chronological OHLC rows: {len(data)}")

    train_end = int(len(data) * 0.70)
    validation_end = int(len(data) * 0.85)
    # Purge the last row before each boundary because its next-candle target
    # overlaps the first candle in the following split.
    train = data.iloc[:train_end - 1]
    train_validation = data.iloc[:validation_end - 1]
    validation = data.iloc[train_end:validation_end - 1]
    test = data.iloc[validation_end:].copy()
    x_train = train[FEATURES]
    y_train = train[TARGETS]
    x_validation = validation[FEATURES]
    y_validation = validation[TARGETS].to_numpy()

    baseline = make_model("persistence_baseline")
    baseline.fit(x_train, y_train)
    baseline_validation = baseline.predict(x_validation)
    forest = make_model("random_forest_regressor")
    forest.fit(x_train, y_train)
    forest_validation = forest.predict(x_validation)

    validation_close = validation["close"].to_numpy()
    baseline_validation = constrain_ohlc_returns(baseline_validation, validation_close)
    forest_validation = constrain_ohlc_returns(forest_validation, validation_close)
    validation_scores = {
        "persistence_baseline": float(np.mean(np.abs(y_validation - baseline_validation))) * 10_000,
        "random_forest_regressor": float(np.mean(np.abs(y_validation - forest_validation))) * 10_000,
    }
    selected_kind = min(validation_scores, key=validation_scores.get)
    selected = make_model(selected_kind)
    selected.fit(train_validation[FEATURES], train_validation[TARGETS])

    x_test = test[FEATURES]
    y_test = test[TARGETS].to_numpy()
    test_predictions = constrain_ohlc_returns(
        selected.predict(x_test), test["close"].to_numpy()
    )
    test_metrics = metric_rows(selected_kind, y_test, test_predictions)
    baseline_test = make_model("persistence_baseline")
    baseline_test.fit(train_validation[FEATURES], train_validation[TARGETS])
    baseline_test_predictions = constrain_ohlc_returns(
        baseline_test.predict(x_test), test["close"].to_numpy()
    )
    test_metrics.extend(metric_rows("persistence_baseline", y_test, baseline_test_predictions))

    output = test[["timestamp", "close", *TARGET_PRICES]].copy()
    predicted_price_columns = [
        "predicted_next_open", "predicted_next_high",
        "predicted_next_low", "predicted_next_close",
    ]
    for index, (target, predicted_price_column) in enumerate(zip(TARGETS, predicted_price_columns)):
        output[f"predicted_{target}"] = test_predictions[:, index]
        output[predicted_price_column] = (
            test["close"].to_numpy() * (1 + test_predictions[:, index])
        )
    # Project each predicted OHLC set back to a valid candle ordering.
    output["predicted_next_high"] = output[
        ["predicted_next_open", "predicted_next_high", "predicted_next_low", "predicted_next_close"]
    ].max(axis=1)
    output["predicted_next_low"] = output[
        ["predicted_next_open", "predicted_next_high", "predicted_next_low", "predicted_next_close"]
    ].min(axis=1)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(PREDICTIONS_PATH, index=False)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_document = {
        "selection_metric": "mean MAE across next-open/high/low/close return targets in basis points",
        "validation_rows": len(validation),
        "purged_boundary_rows": 2,
        "validation_period": [validation["timestamp"].min().isoformat(), validation["timestamp"].max().isoformat()],
        "validation_mean_mae_bps": validation_scores,
        "selected_model": selected_kind,
        "test_rows": len(test),
        "test_period": [test["timestamp"].min().isoformat(), test["timestamp"].max().isoformat()],
        "test_metrics": test_metrics,
        "note": "Regression estimates are uncertain. Test results are historical and do not guarantee live accuracy.",
    }
    METRICS_PATH.write_text(json.dumps(metrics_document, indent=2), encoding="utf-8")

    # Refit the selected model on all labeled rows for live inference, after scoring
    # the chronological test period and saving its out-of-sample metrics.
    production_model = make_model(selected_kind)
    production_model.fit(data[FEATURES], data[TARGETS])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(production_model, MODEL_PATH)
    metadata = {
        "model_version": "futures-ohlc-rf-v1" if selected_kind == "random_forest_regressor" else "futures-ohlc-persistence-v1",
        "model_type": selected_kind,
        "target_type": "next-hour OHLC returns relative to current completed close",
        "features": FEATURES,
        "targets": TARGETS,
        "target_price_columns": TARGET_PRICES,
        "parameters": RANDOM_FOREST_PARAMETERS if selected_kind == "random_forest_regressor" else {"strategy": "constant zero return"},
        "training_rows": len(data),
        "validation_boundary_purge_rows": 2,
        "training_start": data["timestamp"].min().isoformat(),
        "training_end": data["timestamp"].max().isoformat(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_data_path": DATA_PATH.as_posix(),
        "training_data_sha256": sha256_file(DATA_PATH),
        "validation_mean_mae_bps": validation_scores,
        "test_metrics_path": METRICS_PATH.as_posix(),
        "usage": "Research estimates only; predicted OHLC prices are uncertain and not a guarantee.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Usable hourly examples: {len(data):,}")
    print(f"Train rows: {len(train):,}; validation rows: {len(validation):,}; final test rows: {len(test):,}")
    print("Purged boundary rows: 2 (one before validation and one before final test)")
    print("Validation mean MAE (basis points; lower is better):")
    for name, value in validation_scores.items():
        print(f"  {name}: {value:.3f}")
    print(f"Selected on validation only: {selected_kind}")
    print("Final chronological test MAE (basis points; lower is better):")
    print(pd.DataFrame(test_metrics).pivot(index="target", columns="model", values="mae_basis_points").round(3).to_string())
    print(f"Saved test metrics to: {METRICS_PATH}")
    print(f"Saved test predictions to: {PREDICTIONS_PATH}")
    print(f"Saved production model to: {MODEL_PATH}")
    print("The test set was scored once; production artifact is then refit on all labeled rows.")


if __name__ == "__main__":
    main()
