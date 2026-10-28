"""Build the advanced feature set from Binance USDⓈ-M candles + funding data."""

from pathlib import Path

import pandas as pd

from src.bitcoin_ml.features.build_advanced_features import build_advanced_features


INPUT_PATH = Path("data/raw/btcusdt_usdm_1h.csv")
FUNDING_PATH = Path("data/raw/btcusdt_usdm_funding.csv")
OUTPUT_PATH = Path("data/processed/btcusdt_usdm_1h_advanced_features.csv")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Run download_binance_usdm.py first: {INPUT_PATH}")

    candles = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])

    funding = None
    if FUNDING_PATH.exists():
        funding = pd.read_csv(FUNDING_PATH, parse_dates=["timestamp"])
        print(f"Loaded funding data: {len(funding):,} settlements")
    else:
        print("No funding data found; funding features will be NaN.")

    dataset = build_advanced_features(candles, funding=funding)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    # Count how many features are non-NaN in the middle of the dataset.
    feature_cols = [c for c in dataset.columns if c not in [
        "timestamp", "open", "high", "low", "close", "volume",
        "next_return", "target_up", "funding_rate",
    ]]
    mid = len(dataset) // 2
    non_null = dataset.iloc[mid][feature_cols].notna().sum()
    print(f"Input futures candles: {len(candles):,}")
    print(f"Output feature rows: {len(dataset):,}")
    print(f"Total feature columns: {len(feature_cols)}")
    print(f"Non-null features at midpoint: {non_null}")
    print(f"Saved advanced feature dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
