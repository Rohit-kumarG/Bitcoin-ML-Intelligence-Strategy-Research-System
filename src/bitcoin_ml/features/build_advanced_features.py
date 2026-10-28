"""Advanced past-only feature engineering for maximum predictive accuracy.

Adds 50+ features across several families:
- Technical indicators (RSI, MACD, Bollinger, ATR, Stochastic, Williams%R, CCI, MFI)
- Multi-timeframe features (4h, 12h, 24h rolling aggregations)
- Funding rate features (raw, rolling avg, z-score, settlement countdown)
- Cyclical time features (hour-of-day, day-of-week sinusoidal encoding)
- Momentum oscillators and trend-strength signals
- Volatility regime features (ATR ratio, Parkinson, realized vs implied)
- Microstructure features (volume profile, price efficiency)

Every feature uses only past and current data — no lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: safe rolling that respects consecutive-hour gaps
# ---------------------------------------------------------------------------

def _consecutive_mask(timestamps: pd.Series, window: int) -> pd.Series:
    """True where the last `window` hourly steps are all consecutive."""
    one_hour = timestamps.diff() == pd.Timedelta(hours=1)
    return (
        one_hour.astype(int)
        .rolling(window=window, min_periods=window)
        .sum()
        .eq(window)
    )


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    sma = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = (upper - lower) / sma.replace(0, np.nan)
    return pct_b, bandwidth


# ---------------------------------------------------------------------------
# ATR (Average True Range)
# ---------------------------------------------------------------------------

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Stochastic Oscillator
# ---------------------------------------------------------------------------

def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    k = 100.0 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d


# ---------------------------------------------------------------------------
# Williams %R
# ---------------------------------------------------------------------------

def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    return -100.0 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)


# ---------------------------------------------------------------------------
# CCI (Commodity Channel Index)
# ---------------------------------------------------------------------------

def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    mad = tp.rolling(window=period, min_periods=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad).replace(0, np.nan)


# ---------------------------------------------------------------------------
# Money Flow Index (volume-weighted RSI)
# ---------------------------------------------------------------------------

def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    mf = tp * volume
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0.0).rolling(window=period, min_periods=period).sum()
    neg_mf = mf.where(delta <= 0, 0.0).rolling(window=period, min_periods=period).sum()
    mfi = 100.0 - 100.0 / (1.0 + pos_mf / neg_mf.replace(0, np.nan))
    return mfi


# ---------------------------------------------------------------------------
# Parkinson volatility (high-low based)
# ---------------------------------------------------------------------------

def _parkinson_volatility(high: pd.Series, low: pd.Series, period: int = 24) -> pd.Series:
    log_hl = np.log(high / low.replace(0, np.nan))
    return np.sqrt(
        (1.0 / (4.0 * np.log(2.0)))
        * (log_hl ** 2).rolling(window=period, min_periods=period).mean()
    )


# ---------------------------------------------------------------------------
# Volume-weighted average price ratio
# ---------------------------------------------------------------------------

def _vwap_ratio(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 24) -> pd.Series:
    typical_price = (high + low + close) / 3.0
    cum_tpv = (typical_price * volume).rolling(window=period, min_periods=period).sum()
    cum_vol = volume.rolling(window=period, min_periods=period).sum()
    vwap = cum_tpv / cum_vol.replace(0, np.nan)
    return close / vwap.replace(0, np.nan) - 1.0


# ---------------------------------------------------------------------------
# Price efficiency ratio (Kaufman-style)
# ---------------------------------------------------------------------------

def _efficiency_ratio(close: pd.Series, period: int = 10) -> pd.Series:
    direction = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(window=period, min_periods=period).sum()
    return direction / volatility.replace(0, np.nan)


# ===================================================================
# MAIN BUILDER
# ===================================================================

def build_advanced_features(
    candles: pd.DataFrame,
    funding: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a comprehensive feature set from OHLCV candles and optional funding data.

    Parameters
    ----------
    candles : pd.DataFrame
        Must contain columns: timestamp, open, high, low, close, volume.
    funding : pd.DataFrame or None
        Optional. Must contain columns: timestamp, funding_rate.

    Returns
    -------
    pd.DataFrame
        The input candles augmented with all advanced features and the
        next-hour target columns (next_return, target_up).
    """
    df = candles.copy()

    # Validate required columns.
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    one_hour_step = df["timestamp"].diff() == pd.Timedelta(hours=1)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    open_ = df["open"]

    # ====================================================================
    # 1. BASIC RETURN FEATURES (same as original)
    # ====================================================================
    for hours in [1, 3, 6, 12, 24, 48]:
        prev_close = close.shift(hours)
        elapsed = df["timestamp"] - df["timestamp"].shift(hours)
        df[f"return_{hours}h"] = (close / prev_close - 1).where(
            elapsed == pd.Timedelta(hours=hours)
        )
    df["return_1h"] = df["return_1h"].where(one_hour_step)

    # ====================================================================
    # 2. VOLUME FEATURES
    # ====================================================================
    df["volume_log_change_1h"] = (
        np.log1p(volume) - np.log1p(volume.shift(1))
    ).where(one_hour_step)

    # Rolling volume metrics.
    for w in [6, 12, 24]:
        rolling_vol = volume.rolling(window=w, min_periods=w).mean()
        df[f"volume_ratio_{w}h"] = volume / rolling_vol.replace(0, np.nan)
        mask = _consecutive_mask(df["timestamp"], w)
        df.loc[~mask, f"volume_ratio_{w}h"] = np.nan

    # Volume trend (slope of log volume over 24 hours).
    df["volume_trend_24h"] = (
        np.log1p(volume)
        .rolling(window=24, min_periods=24)
        .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
    )
    df.loc[~_consecutive_mask(df["timestamp"], 24), "volume_trend_24h"] = np.nan

    # ====================================================================
    # 3. CANDLE STRUCTURE FEATURES (same as original, plus extras)
    # ====================================================================
    df["candle_body_pct"] = (close - open_) / open_
    df["candle_range_pct"] = (high - low) / open_
    df["upper_wick_pct"] = (high - df[["open", "close"]].max(axis=1)) / open_
    df["lower_wick_pct"] = (df[["open", "close"]].min(axis=1) - low) / open_

    # Body-to-range ratio (how decisive is the candle).
    candle_range = high - low
    df["body_to_range"] = (close - open_).abs() / candle_range.replace(0, np.nan)

    # Gap: difference between this candle's open and previous close.
    prev_close_1 = close.shift(1)
    df["gap_pct"] = ((open_ - prev_close_1) / prev_close_1).where(one_hour_step)

    # ====================================================================
    # 4. TECHNICAL INDICATORS
    # ====================================================================
    # RSI at multiple periods.
    for period in [6, 14, 24]:
        df[f"rsi_{period}"] = _rsi(close, period)

    # MACD.
    macd_line, macd_signal, macd_hist = _macd(close)
    df["macd_line"] = macd_line / close  # normalize
    df["macd_signal"] = macd_signal / close
    df["macd_histogram"] = macd_hist / close

    # Bollinger Bands.
    pct_b, bandwidth = _bollinger(close)
    df["bollinger_pct_b"] = pct_b
    df["bollinger_bandwidth"] = bandwidth

    # ATR normalized by close.
    atr_14 = _atr(high, low, close, 14)
    df["atr_14_pct"] = atr_14 / close
    atr_6 = _atr(high, low, close, 6)
    df["atr_6_pct"] = atr_6 / close

    # ATR ratio (short / long) — volatility expansion/contraction signal.
    df["atr_ratio_6_14"] = atr_6 / atr_14.replace(0, np.nan)

    # Stochastic oscillator.
    stoch_k, stoch_d = _stochastic(high, low, close)
    df["stochastic_k"] = stoch_k
    df["stochastic_d"] = stoch_d

    # Williams %R.
    df["williams_r_14"] = _williams_r(high, low, close)

    # CCI.
    df["cci_20"] = _cci(high, low, close)

    # MFI (volume-weighted RSI).
    df["mfi_14"] = _mfi(high, low, close, volume)

    # ====================================================================
    # 5. VOLATILITY FEATURES
    # ====================================================================
    # Rolling realized volatility (standard deviation of 1h returns).
    for w in [12, 24, 48]:
        df[f"volatility_{w}h"] = df["return_1h"].rolling(
            window=w, min_periods=w
        ).std()
        mask = _consecutive_mask(df["timestamp"], w)
        df.loc[~mask, f"volatility_{w}h"] = np.nan

    # Parkinson volatility (uses high/low — more efficient estimator).
    df["parkinson_vol_24h"] = _parkinson_volatility(high, low, 24)

    # Volatility ratio: short / long — detects volatility regime changes.
    df["vol_ratio_12_48"] = df["volatility_12h"] / df["volatility_48h"].replace(0, np.nan)

    # VWAP ratio.
    df["vwap_ratio_24h"] = _vwap_ratio(high, low, close, volume, 24)

    # Price efficiency ratio.
    df["efficiency_ratio_10"] = _efficiency_ratio(close, 10)

    # ====================================================================
    # 6. MULTI-TIMEFRAME FEATURES (rolling aggregations)
    # ====================================================================
    # 4-hour and 12-hour trend.
    for w in [4, 12]:
        df[f"trend_{w}h"] = (close / close.shift(w) - 1).where(
            (df["timestamp"] - df["timestamp"].shift(w)) == pd.Timedelta(hours=w)
        )

    # Distance from rolling high/low (support/resistance proxy).
    for w in [24, 72]:
        rolling_high = high.rolling(window=w, min_periods=w).max()
        rolling_low = low.rolling(window=w, min_periods=w).min()
        range_ = rolling_high - rolling_low
        df[f"dist_from_high_{w}h"] = (rolling_high - close) / range_.replace(0, np.nan)
        df[f"dist_from_low_{w}h"] = (close - rolling_low) / range_.replace(0, np.nan)

    # EMA crossover signals.
    ema_9 = close.ewm(span=9, min_periods=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, min_periods=21, adjust=False).mean()
    ema_50 = close.ewm(span=50, min_periods=50, adjust=False).mean()
    df["ema_9_21_spread"] = (ema_9 - ema_21) / close
    df["ema_21_50_spread"] = (ema_21 - ema_50) / close
    df["price_vs_ema_50"] = (close - ema_50) / close

    # ====================================================================
    # 7. CYCLICAL TIME FEATURES
    # ====================================================================
    hour = df["timestamp"].dt.hour
    day_of_week = df["timestamp"].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    # Session indicators: Asian (00-08 UTC), European (08-16 UTC), American (16-24 UTC).
    df["session_asian"] = ((hour >= 0) & (hour < 8)).astype(float)
    df["session_european"] = ((hour >= 8) & (hour < 16)).astype(float)
    df["session_american"] = ((hour >= 16) & (hour < 24)).astype(float)

    # ====================================================================
    # 8. FUNDING RATE FEATURES (if funding data is available)
    # ====================================================================
    if funding is not None and not funding.empty:
        fund = funding.copy()
        fund["timestamp"] = pd.to_datetime(fund["timestamp"], utc=True)
        fund = fund.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

        # Forward-fill funding rate to hourly resolution.
        # Funding settles every 8 hours; fill forward to each hour.
        df = df.merge(
            fund[["timestamp", "funding_rate"]],
            on="timestamp",
            how="left",
        )
        df["funding_rate"] = df["funding_rate"].ffill()

        # Rolling funding averages.
        for w in [3, 8, 24]:
            df[f"funding_rate_ma_{w}"] = df["funding_rate"].rolling(
                window=w, min_periods=1
            ).mean()

        # Funding rate z-score (deviation from recent norm).
        funding_mean_72 = df["funding_rate"].rolling(window=72, min_periods=24).mean()
        funding_std_72 = df["funding_rate"].rolling(window=72, min_periods=24).std()
        df["funding_zscore"] = (df["funding_rate"] - funding_mean_72) / funding_std_72.replace(0, np.nan)

        # Hours until next funding settlement (0, 8, 16 UTC).
        df["hours_to_settlement"] = (8 - (hour % 8)) % 8
        df["settlement_imminent"] = (df["hours_to_settlement"] <= 1).astype(float)

        # Funding rate direction (positive = longs pay = crowded long).
        df["funding_positive"] = (df["funding_rate"] > 0).astype(float)

        # Funding pressure: extreme funding often precedes reversals.
        funding_abs_mean = df["funding_rate"].abs().rolling(window=24, min_periods=8).mean()
        df["funding_pressure"] = df["funding_rate"].abs() / funding_abs_mean.replace(0, np.nan)
    else:
        # Add placeholder columns so downstream code sees a consistent schema.
        for col in [
            "funding_rate", "funding_rate_ma_3", "funding_rate_ma_8",
            "funding_rate_ma_24", "funding_zscore", "hours_to_settlement",
            "settlement_imminent", "funding_positive", "funding_pressure",
        ]:
            df[col] = np.nan

    # ====================================================================
    # 9. MOMENTUM FEATURES
    # ====================================================================
    # Rate of change (ROC).
    for w in [6, 12, 24]:
        df[f"roc_{w}h"] = (close / close.shift(w) - 1).where(
            (df["timestamp"] - df["timestamp"].shift(w)) == pd.Timedelta(hours=w)
        )

    # Consecutive up/down candles (streak).
    up_candle = (close > open_).astype(int)
    down_candle = (close < open_).astype(int)
    # Count consecutive bullish candles.
    df["consecutive_up"] = up_candle * (
        up_candle.groupby((up_candle != up_candle.shift()).cumsum()).cumcount() + 1
    )
    df["consecutive_down"] = down_candle * (
        down_candle.groupby((down_candle != down_candle.shift()).cumsum()).cumcount() + 1
    )

    # ====================================================================
    # 10. INTERACTION / COMPOSITE FEATURES
    # ====================================================================
    # RSI * volume ratio — high RSI + high volume = stronger signal.
    df["rsi_volume_interaction"] = df["rsi_14"] * df.get("volume_ratio_24h", 1.0)

    # Bollinger squeeze + low ATR = breakout imminent.
    df["squeeze_score"] = df["bollinger_bandwidth"] * df["atr_14_pct"]

    # Trend-volatility interaction.
    df["trend_vol_interaction"] = df.get("return_24h", 0.0) * df.get("volatility_24h", 0.0)

    # ====================================================================
    # 11. TARGET VARIABLES (same logic as original)
    # ====================================================================
    next_timestamp = df["timestamp"].shift(-1)
    next_close = close.shift(-1)
    next_is_one_hour = (next_timestamp - df["timestamp"]) == pd.Timedelta(hours=1)

    df["next_return"] = (next_close / close - 1).where(next_is_one_hour)
    df["target_up"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    valid_target = df["next_return"].notna()
    df.loc[valid_target, "target_up"] = (
        df.loc[valid_target, "next_return"] > 0
    ).astype("Int64")

    return df


# ---------------------------------------------------------------------------
# Feature name lists for downstream consumers
# ---------------------------------------------------------------------------

BASIC_FEATURES = [
    "return_1h", "return_3h", "return_6h", "return_12h", "return_24h", "return_48h",
    "volatility_12h", "volatility_24h", "volatility_48h",
    "volume_log_change_1h", "candle_body_pct", "candle_range_pct",
    "upper_wick_pct", "lower_wick_pct",
]

TECHNICAL_FEATURES = [
    "rsi_6", "rsi_14", "rsi_24",
    "macd_line", "macd_signal", "macd_histogram",
    "bollinger_pct_b", "bollinger_bandwidth",
    "atr_14_pct", "atr_6_pct", "atr_ratio_6_14",
    "stochastic_k", "stochastic_d",
    "williams_r_14", "cci_20", "mfi_14",
]

VOLUME_FEATURES = [
    "volume_ratio_6h", "volume_ratio_12h", "volume_ratio_24h",
    "volume_trend_24h",
]

STRUCTURE_FEATURES = [
    "body_to_range", "gap_pct",
]

VOLATILITY_FEATURES = [
    "parkinson_vol_24h", "vol_ratio_12_48",
    "vwap_ratio_24h", "efficiency_ratio_10",
]

MULTI_TIMEFRAME_FEATURES = [
    "trend_4h", "trend_12h",
    "dist_from_high_24h", "dist_from_low_24h",
    "dist_from_high_72h", "dist_from_low_72h",
    "ema_9_21_spread", "ema_21_50_spread", "price_vs_ema_50",
]

CYCLICAL_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "session_asian", "session_european", "session_american",
]

FUNDING_FEATURES = [
    "funding_rate", "funding_rate_ma_3", "funding_rate_ma_8",
    "funding_rate_ma_24", "funding_zscore", "hours_to_settlement",
    "settlement_imminent", "funding_positive", "funding_pressure",
]

MOMENTUM_FEATURES = [
    "roc_6h", "roc_12h", "roc_24h",
    "consecutive_up", "consecutive_down",
]

INTERACTION_FEATURES = [
    "rsi_volume_interaction", "squeeze_score", "trend_vol_interaction",
]

ALL_ADVANCED_FEATURES = (
    BASIC_FEATURES
    + TECHNICAL_FEATURES
    + VOLUME_FEATURES
    + STRUCTURE_FEATURES
    + VOLATILITY_FEATURES
    + MULTI_TIMEFRAME_FEATURES
    + CYCLICAL_FEATURES
    + FUNDING_FEATURES
    + MOMENTUM_FEATURES
    + INTERACTION_FEATURES
)
