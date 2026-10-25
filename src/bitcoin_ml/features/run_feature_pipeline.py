from pathlib import Path

import pandas as pd

from src.bitcoin_ml.features.build_features import build_features


def main() -> None:
    input_path = Path("data/raw/btcusdt_1h.csv")
    output_path = Path("data/processed/btcusdt_1h_features.csv")

    candles = pd.read_csv(input_path, parse_dates=["timestamp"])
    dataset = build_features(candles)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    print(f"Input candles: {len(candles):,}")
    print(f"Output rows: {len(dataset):,}")
    print(f"Saved feature dataset to: {output_path}")


if __name__ == "__main__":
    main()