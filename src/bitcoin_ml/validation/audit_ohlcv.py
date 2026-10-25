from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/btcusdt_1h.csv")
EXPECTED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Rows (hourly candles): {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    print(f"Duplicate timestamps: {df['timestamp'].duplicated().sum()}")
    print(f"Missing values per column:\n{df.isna().sum()}")
    print(f"Zero-volume candles: {(df['volume'] == 0).sum()}")

    invalid_ohlc = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        | (df["high"] < df["low"])
        | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (df["volume"] < 0)
    )
    print(f"Invalid OHLCV candles: {invalid_ohlc.sum()}")
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    gaps = df["timestamp"].diff()
    gap_rows = df.loc[gaps > pd.Timedelta(hours=1), ["timestamp"]].copy()
    gap_rows["gap_duration"] = gaps[gaps > pd.Timedelta(hours=1)].values

    print(f"Missing hourly intervals: {len(gap_rows)}")
    if not gap_rows.empty:
        print(gap_rows.to_string(index=False))


if __name__ == "__main__":
    main()