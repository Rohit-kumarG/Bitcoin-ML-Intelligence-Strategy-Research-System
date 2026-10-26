"""Build the existing past-only feature set from Binance USDⓈ-M candles."""

from pathlib import Path

import pandas as pd

from src.bitcoin_ml.features.build_features import build_features


INPUT_PATH = Path("data/raw/btcusdt_usdm_1h.csv")
OUTPUT_PATH = Path("data/processed/btcusdt_usdm_1h_features.csv")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Run download_binance_usdm.py first: {INPUT_PATH}")
    candles = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    dataset = build_features(candles)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Input futures candles: {len(candles):,}")
    print(f"Output feature rows: {len(dataset):,}")
    print(f"Saved futures feature dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
