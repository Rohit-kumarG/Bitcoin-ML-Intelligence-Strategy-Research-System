"""Monitor OOF model performance and feature drift across chronological folds."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
)


PREDICTIONS_PATH = Path("artifacts/metrics/futures_walk_forward_probabilities.csv")
FEATURES_PATH = Path("data/processed/btcusdt_usdm_1h_features.csv")
MONTHLY_PATH = Path("artifacts/metrics/futures_monitoring_monthly.csv")
DRIFT_PATH = Path("artifacts/metrics/futures_feature_drift.csv")
REPORT_PATH = Path("artifacts/metrics/futures_monitoring_report.md")
FEATURES = [
    "return_1h", "return_3h", "return_6h", "return_24h", "volatility_24h",
    "volume_log_change_1h", "candle_body_pct", "candle_range_pct",
    "upper_wick_pct", "lower_wick_pct",
]


def population_stability_index(reference: pd.Series, current: pd.Series) -> float:
    """Compare current feature distribution with reference using reference quantile bins."""
    reference = reference.replace([np.inf, -np.inf], np.nan).dropna()
    current = current.replace([np.inf, -np.inf], np.nan).dropna()
    if reference.empty or current.empty:
        return float("nan")

    _, edges = pd.qcut(reference, q=10, retbins=True, duplicates="drop")
    edges[0], edges[-1] = -np.inf, np.inf
    if len(edges) < 3:
        return 0.0

    reference_bins = pd.cut(reference, bins=edges, include_lowest=True).value_counts(
        sort=False, normalize=True
    )
    current_bins = pd.cut(current, bins=edges, include_lowest=True).value_counts(
        sort=False, normalize=True
    )
    epsilon = 1e-6
    ref_share = reference_bins.to_numpy() + epsilon
    cur_share = current_bins.to_numpy() + epsilon
    ref_share /= ref_share.sum()
    cur_share /= cur_share.sum()
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def main() -> None:
    for path in [PREDICTIONS_PATH, FEATURES_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required monitoring input is missing: {path}")

    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["timestamp"])
    features = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])
    data = predictions.merge(
        features[["timestamp", *FEATURES]],
        on="timestamp",
        how="left",
        validate="one_to_one",
    ).dropna(subset=["actual_up", "probability_up", *FEATURES]).copy()
    data["actual_up"] = data["actual_up"].astype(int)
    data["month"] = data["timestamp"].dt.strftime("%Y-%m")
    data["predicted_up"] = (data["probability_up"] >= 0.5).astype(int)
    data["confidence"] = data["probability_up"].where(
        data["predicted_up"] == 1, 1 - data["probability_up"]
    )

    monthly_rows = []
    for month, group in data.groupby("month", sort=True):
        monthly_rows.append(
            {
                "month": month,
                "rows": len(group),
                "up_rate": group["actual_up"].mean(),
                "mean_probability_up": group["probability_up"].mean(),
                "accuracy": (group["predicted_up"] == group["actual_up"]).mean(),
                "balanced_accuracy": balanced_accuracy_score(
                    group["actual_up"], group["predicted_up"]
                ),
                "brier_score": brier_score_loss(
                    group["actual_up"], group["probability_up"]
                ),
                "log_loss": log_loss(
                    group["actual_up"], group["probability_up"], labels=[0, 1]
                ),
                "high_confidence_share_60pct": (group["confidence"] >= 0.60).mean(),
            }
        )
    monthly = pd.DataFrame(monthly_rows)

    first_fold = data[data["fold"] == data["fold"].min()]
    drift_rows = []
    for fold, group in data.groupby("fold", sort=True):
        for feature in FEATURES:
            drift_rows.append(
                {
                    "reference_fold": int(data["fold"].min()),
                    "fold": int(fold),
                    "feature": feature,
                    "reference_mean": first_fold[feature].mean(),
                    "fold_mean": group[feature].mean(),
                    "reference_median": first_fold[feature].median(),
                    "fold_median": group[feature].median(),
                    "psi_vs_first_validation_fold": population_stability_index(
                        first_fold[feature], group[feature]
                    ),
                }
            )
    drift = pd.DataFrame(drift_rows)

    MONTHLY_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(MONTHLY_PATH, index=False)
    drift.to_csv(DRIFT_PATH, index=False)

    fold_performance = data.groupby("fold").agg(
        rows=("actual_up", "size"),
        accuracy=("predicted_up", lambda values: (values == data.loc[values.index, "actual_up"]).mean()),
        mean_probability_up=("probability_up", "mean"),
        observed_up_rate=("actual_up", "mean"),
    )
    report_lines = [
        "# Futures model monitoring report",
        "",
        f"OOF validation rows analyzed: {len(data):,}.",
        f"Validation range: {data['timestamp'].min()} to {data['timestamp'].max()}.",
        "The final 15% test period is excluded.",
        "",
        "## Performance by walk-forward fold",
        "",
        "```text",
        fold_performance.round(4).to_string(),
        "```",
        "",
        "## PSI interpretation",
        "",
        "PSI compares each fold's feature distribution with the first validation fold. "
        "It is a monitoring signal, not proof of model failure or a reason to retrain by itself. "
        "Common rough guides are PSI below 0.10 (small change), 0.10-0.25 (noticeable), "
        "and above 0.25 (large change); these are heuristics, not universal rules.",
        "",
        "## Limits",
        "",
        "Monthly windows can be noisy. Review performance and drift together, and decide "
        "retraining policy using development data only. Do not use the reserved final test "
        "to tune thresholds or trigger a repeated experiment.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"OOF validation rows monitored: {len(data):,}")
    print(f"Validation range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print("Final test: still reserved and untouched")
    print("\nMonthly prediction monitoring (first and last 6 months):")
    print(pd.concat([monthly.head(6), monthly.tail(6)]).round(4).to_string(index=False))
    print("\nMean feature PSI by fold vs first validation fold:")
    print(drift.groupby("fold")["psi_vs_first_validation_fold"].mean().round(4).to_string())
    print(f"\nSaved monthly metrics to: {MONTHLY_PATH}")
    print(f"Saved feature drift metrics to: {DRIFT_PATH}")
    print(f"Saved monitoring report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
