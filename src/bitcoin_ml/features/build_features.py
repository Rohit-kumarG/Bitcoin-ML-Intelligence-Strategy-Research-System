import pandas as pd


def build_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Create past-only features and the next-hour direction target."""
    df = candles.copy()

    # Check required input columns before using them.
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Make timestamps readable as UTC datetimes and sort candles by time.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # True when this row is exactly one hour after the previous row.
    one_hour_step = df["timestamp"].diff() == pd.Timedelta(hours=1)

    # Past returns: compare this close with the close 1, 3, 6, or 24 hours ago.
    for hours in [1, 3, 6, 24]:
        previous_close = df["close"].shift(hours)
        elapsed = df["timestamp"] - df["timestamp"].shift(hours)

        df[f"return_{hours}h"] = (
            df["close"] / previous_close - 1
        ).where(elapsed == pd.Timedelta(hours=hours))

    # A one-hour return is valid only across consecutive hourly candles.
    df["return_1h"] = df["return_1h"].where(one_hour_step)

    # Standard deviation of the latest 24 one-hour returns.
    # Keep it only if all 24 hourly transitions are consecutive.
    df["volatility_24h"] = df["return_1h"].rolling(
        window=24,
        min_periods=24,
    ).std()

    consecutive_24h = (
        one_hour_step.astype(int)
        .rolling(window=24, min_periods=24)
        .sum()
        .eq(24)
    )
    df.loc[~consecutive_24h, "volatility_24h"] = float("nan")

    # Next-hour outcome: compare the next close with this row's close.
    next_timestamp = df["timestamp"].shift(-1)
    next_close = df["close"].shift(-1)
    next_is_one_hour = (
        next_timestamp - df["timestamp"]
    ) == pd.Timedelta(hours=1)

    df["next_return"] = (
        next_close / df["close"] - 1
    ).where(next_is_one_hour)

    # 1 = next-hour return is positive; 0 = zero or negative.
    # Rows without a valid next-hour outcome stay missing.
    df["target_up"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    valid_target = df["next_return"].notna()
    df.loc[valid_target, "target_up"] = (
        df.loc[valid_target, "next_return"] > 0
    ).astype("Int64")

    return df