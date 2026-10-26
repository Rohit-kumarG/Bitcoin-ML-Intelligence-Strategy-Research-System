"""Flag unusual market observations with out-of-sample Isolation Forest."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import TimeSeriesSplit


DATA_PATH = Path("data/processed/btcusdt_1h_features.csv")
OUTPUT_PATH = Path("artifacts/metrics/anomaly_validation_predictions.csv")
TOP_EVENTS_PATH = Path("artifacts/metrics/anomaly_top_events.csv")
ANOMALY_FEATURES = [
    "return_1h",
    "return_24h",
    "volatility_24h",
    "volume_log_change_1h",
    "candle_range_pct",
]
TARGET = "target_up"
N_SPLITS = 5
VALIDATION_HOURS = 24 * 182
FINAL_TEST_FRACTION = 0.15
CONTAMINATION = 0.01


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.dropna(subset=ANOMALY_FEATURES + [TARGET]).sort_values("timestamp").reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)

    # Keep the newest 15% out of all research in this phase.
    development_size = int(len(df) * (1 - FINAL_TEST_FRACTION))
    development = df.iloc[:development_size].copy()

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=VALIDATION_HOURS,
        gap=1,
    )

    prediction_frames = []
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(development), start=1
    ):
        train = development.iloc[train_indices]
        validation = development.iloc[validation_indices]

        detector = IsolationForest(
            n_estimators=300,
            contamination=CONTAMINATION,
            random_state=42,
            n_jobs=-1,
        )
        # Fit only on past market features. The future target is never an input.
        detector.fit(train[ANOMALY_FEATURES])

        scored = validation[
            ["timestamp", *ANOMALY_FEATURES, "next_return", TARGET]
        ].copy()
        scored.insert(0, "fold", fold)
        scored["anomaly_score"] = detector.decision_function(
            validation[ANOMALY_FEATURES]
        )
        scored["is_anomaly"] = detector.predict(
            validation[ANOMALY_FEATURES]
        ) == -1
        prediction_frames.append(scored)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_PATH, index=False)

    summary = predictions.groupby("is_anomaly").agg(
        observations=("timestamp", "size"),
        mean_abs_hourly_return=("return_1h", lambda values: values.abs().mean()),
        mean_abs_volume_log_change=(
            "volume_log_change_1h", lambda values: values.abs().mean()
        ),
        mean_next_return=("next_return", "mean"),
        next_up_rate=(TARGET, "mean"),
    )
    summary.index = summary.index.map({False: "normal", True: "flagged_anomaly"})

    top_events = predictions.nsmallest(20, "anomaly_score")
    top_events.to_csv(TOP_EVENTS_PATH, index=False)

    print(f"Usable rows: {len(df):,}")
    print(f"Development rows evaluated: {len(predictions):,}")
    print(f"Final test rows reserved: {len(df) - development_size:,}")
    print(f"Features used (target excluded): {', '.join(ANOMALY_FEATURES)}")
    print(f"Flagging threshold assumption: {CONTAMINATION:.0%} contamination")
    print("\nValidation summary (target shown only for later analysis):")
    print(summary.round(4).to_string())
    print(f"\nSaved validation flags to: {OUTPUT_PATH}")
    print(f"Saved 20 most unusual validation rows to: {TOP_EVENTS_PATH}")


if __name__ == "__main__":
    main()
