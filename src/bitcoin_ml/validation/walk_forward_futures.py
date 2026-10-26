"""Create out-of-fold Random Forest probabilities from futures-native features."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit


DATA_PATH = Path("data/processed/btcusdt_usdm_1h_features.csv")
PROBABILITY_PATH = Path("artifacts/metrics/futures_walk_forward_probabilities.csv")
METRICS_PATH = Path("artifacts/metrics/futures_walk_forward_metrics.csv")
FEATURES = [
    "return_1h", "return_3h", "return_6h", "return_24h", "volatility_24h",
    "volume_log_change_1h", "candle_body_pct", "candle_range_pct",
    "upper_wick_pct", "lower_wick_pct",
]
TARGET = "target_up"
N_SPLITS = 5
VALIDATION_HOURS = 24 * 182
FINAL_TEST_FRACTION = 0.15


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run run_futures_feature_pipeline.py first.")
    data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    data = data.dropna(subset=FEATURES + [TARGET]).sort_values("timestamp").reset_index(drop=True)
    data[TARGET] = data[TARGET].astype(int)

    development_size = int(len(data) * (1 - FINAL_TEST_FRACTION))
    development = data.iloc[:development_size]
    final_test = data.iloc[development_size:]
    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS, test_size=VALIDATION_HOURS, gap=1
    )

    predictions = []
    metrics = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(development), start=1):
        train = development.iloc[train_idx]
        valid = development.iloc[valid_idx]
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train[FEATURES], train[TARGET])
        probability_up = model.predict_proba(valid[FEATURES])[:, 1]
        predicted_up = (probability_up >= 0.5).astype(int)
        metrics.append(
            {
                "fold": fold,
                "train_start": train["timestamp"].min(),
                "train_end": train["timestamp"].max(),
                "validation_start": valid["timestamp"].min(),
                "validation_end": valid["timestamp"].max(),
                "validation_rows": len(valid),
                "accuracy": accuracy_score(valid[TARGET], predicted_up),
                "balanced_accuracy": balanced_accuracy_score(valid[TARGET], predicted_up),
            }
        )
        predictions.extend(
            {
                "fold": fold,
                "timestamp": timestamp,
                "actual_up": int(actual),
                "probability_up": float(probability),
            }
            for timestamp, actual, probability in zip(
                valid["timestamp"], valid[TARGET], probability_up
            )
        )

    prediction_frame = pd.DataFrame(predictions)
    metric_frame = pd.DataFrame(metrics)
    PROBABILITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(PROBABILITY_PATH, index=False)
    metric_frame.to_csv(METRICS_PATH, index=False)

    print(f"Futures feature rows: {len(data):,}")
    print(f"Development rows evaluated: {len(prediction_frame):,}")
    print(f"Final test rows reserved: {len(final_test):,}")
    print("Model: Random Forest; expanding chronological folds; final test untouched")
    print("\nFold metrics:")
    print(metric_frame[["fold", "validation_rows", "accuracy", "balanced_accuracy"]].round(4).to_string(index=False))
    print("\nMean validation balanced accuracy:", round(metric_frame["balanced_accuracy"].mean(), 4))
    print(f"Saved OOF probabilities to: {PROBABILITY_PATH}")
    print(f"Saved fold metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
