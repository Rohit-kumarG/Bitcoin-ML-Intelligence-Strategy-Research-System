"""Compare walk-forward scores when individual feature families are removed."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit


DATA_PATH = Path("data/processed/btcusdt_1h_features.csv")
OUTPUT_PATH = Path("artifacts/metrics/feature_ablation.csv")
FEATURES = [
    "return_1h",
    "return_3h",
    "return_6h",
    "return_24h",
    "volatility_24h",
    "volume_log_change_1h",
    "candle_body_pct",
    "candle_range_pct",
    "upper_wick_pct",
    "lower_wick_pct",
]
FEATURE_FAMILIES = {
    "returns": ["return_1h", "return_3h", "return_6h", "return_24h"],
    "volatility": ["volatility_24h"],
    "volume": ["volume_log_change_1h"],
    "candle_structure": [
        "candle_body_pct",
        "candle_range_pct",
        "upper_wick_pct",
        "lower_wick_pct",
    ],
}
TARGET = "target_up"
N_SPLITS = 5
VALIDATION_HOURS = 24 * 182
FINAL_TEST_FRACTION = 0.15


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.dropna(subset=FEATURES + [TARGET]).sort_values("timestamp").reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)

    # Keep the latest 15% separate for final evaluation.
    development_size = int(len(df) * (1 - FINAL_TEST_FRACTION))
    development = df.iloc[:development_size].copy()

    feature_sets = {"all_features": FEATURES}
    for family_name, family_features in FEATURE_FAMILIES.items():
        feature_sets[f"without_{family_name}"] = [
            feature for feature in FEATURES if feature not in family_features
        ]

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=VALIDATION_HOURS,
        gap=1,
    )

    rows = []
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(development), start=1
    ):
        train = development.iloc[train_indices]
        validation = development.iloc[validation_indices]
        y_train = train[TARGET]
        y_validation = validation[TARGET]

        for experiment_name, selected_features in feature_sets.items():
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(train[selected_features], y_train)
            predictions = model.predict(validation[selected_features])

            rows.append(
                {
                    "fold": fold,
                    "experiment": experiment_name,
                    "removed_family": (
                        "none" if experiment_name == "all_features"
                        else experiment_name.removeprefix("without_")
                    ),
                    "validation_start": validation["timestamp"].min(),
                    "validation_end": validation["timestamp"].max(),
                    "accuracy": accuracy_score(y_validation, predictions),
                    "balanced_accuracy": balanced_accuracy_score(y_validation, predictions),
                    "f1": f1_score(y_validation, predictions, zero_division=0),
                }
            )

    results = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    summary = results.groupby("experiment")[
        ["accuracy", "balanced_accuracy", "f1"]
    ].agg(["mean", "std"])
    print("Experiment: Random Forest, same five chronological validation folds")
    print("Comparison: all features versus removing one feature family at a time")
    print("Final 15% test period remains untouched.")
    print("\nMean and standard deviation across folds:")
    print(summary.round(4).to_string())
    print(f"\nSaved fold-level results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
