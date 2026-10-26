"""Discover volatility regimes on training data and compare validation behavior."""

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/processed/btcusdt_1h_features.csv")
OUTPUT_PATH = Path("artifacts/metrics/regime_validation_metrics.csv")
REGIME_FEATURES = ["return_24h", "volatility_24h"]
MODEL_FEATURES = [
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
TARGET = "target_up"
N_SPLITS = 5
VALIDATION_HOURS = 24 * 182
FINAL_TEST_FRACTION = 0.15
N_REGIMES = 3


def main() -> None:
    required = list(set(REGIME_FEATURES + MODEL_FEATURES + [TARGET, "timestamp"]))
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.dropna(subset=required).sort_values("timestamp").reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)

    # Reserve the newest 15% for final evaluation; folds use the older 85% only.
    development_size = int(len(df) * (1 - FINAL_TEST_FRACTION))
    development = df.iloc[:development_size].copy()
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
        validation = development.iloc[validation_indices].copy()

        # Fit scaling and clusters on this fold's past training data only.
        regime_scaler = StandardScaler()
        train_regime_x = regime_scaler.fit_transform(train[REGIME_FEATURES])
        clusterer = KMeans(n_clusters=N_REGIMES, n_init=10, random_state=42)
        clusterer.fit(train_regime_x)
        validation["cluster_id"] = clusterer.predict(
            regime_scaler.transform(validation[REGIME_FEATURES])
        )

        # Give cluster IDs stable, interpretable names by ranking training volatility.
        centers = regime_scaler.inverse_transform(clusterer.cluster_centers_)
        volatility_order = centers[:, 1].argsort()
        regime_names = {cluster_id: name for name, cluster_id in zip(
            ["low_volatility", "mid_volatility", "high_volatility"],
            volatility_order,
        )}
        validation["regime"] = validation["cluster_id"].map(regime_names)

        # Compare the same fixed Random Forest within the validation regimes.
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train[MODEL_FEATURES], train[TARGET])
        validation["prediction"] = model.predict(validation[MODEL_FEATURES])

        for regime_name, group in validation.groupby("regime", observed=True):
            cluster_id = next(
                cluster_id for cluster_id, name in regime_names.items()
                if name == regime_name
            )
            rows.append(
                {
                    "fold": fold,
                    "regime": regime_name,
                    "validation_rows": len(group),
                    "validation_start": group["timestamp"].min(),
                    "validation_end": group["timestamp"].max(),
                    "train_cluster_mean_return_24h": centers[cluster_id, 0],
                    "train_cluster_mean_volatility_24h": centers[cluster_id, 1],
                    "observed_up_rate": group[TARGET].mean(),
                    "accuracy": accuracy_score(group[TARGET], group["prediction"]),
                    "balanced_accuracy": balanced_accuracy_score(
                        group[TARGET], group["prediction"]
                    ),
                }
            )

    results = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    summary = results.groupby("regime")[[
        "validation_rows",
        "observed_up_rate",
        "accuracy",
        "balanced_accuracy",
    ]].mean()

    print(f"Usable rows: {len(df):,}")
    print(f"Development rows used: {len(development):,}")
    print(f"Final test rows reserved: {len(df) - len(development):,}")
    print("K-Means inputs: trailing 24h return and 24h volatility")
    print("Clusters are fit within each fold's training period only.")
    print("\nMean validation results by volatility-ranked regime:")
    print(summary.round(4).to_string())
    print(f"\nSaved fold-level results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
