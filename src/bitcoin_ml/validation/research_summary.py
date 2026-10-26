"""Consolidate Phase 5-8 validation evidence without touching the final test set."""

from pathlib import Path

import pandas as pd


METRICS_PATH = Path("artifacts/metrics/walk_forward_metrics.csv")
PROBABILITIES_PATH = Path("artifacts/metrics/walk_forward_probabilities.csv")
ABLATION_PATH = Path("artifacts/metrics/feature_ablation.csv")
REGIME_PATH = Path("artifacts/metrics/regime_validation_metrics.csv")
ANOMALY_PATH = Path("artifacts/metrics/anomaly_validation_predictions.csv")
OUTPUT_DIR = Path("artifacts/metrics")
THRESHOLDS = [0.55, 0.60, 0.65, 0.70]


def build_threshold_analysis(probabilities: pd.DataFrame) -> pd.DataFrame:
    """Measure accuracy on predictions where the model's class confidence is high."""
    rows = []
    for model_name, model_rows in probabilities.groupby("model"):
        probability_up = model_rows["probability_up"]
        actual = model_rows["actual_up"].astype(int)
        predicted = (probability_up >= 0.5).astype(int)
        confidence = probability_up.where(predicted.eq(1), 1 - probability_up)

        for threshold in THRESHOLDS:
            selected = confidence >= threshold
            count = int(selected.sum())
            rows.append(
                {
                    "model": model_name,
                    "confidence_threshold": threshold,
                    "covered_rows": count,
                    "coverage_pct": 100 * count / len(model_rows),
                    "directional_accuracy_pct": (
                        100 * (predicted[selected] == actual[selected]).mean()
                        if count else float("nan")
                    ),
                    "mean_confidence_pct": (
                        100 * confidence[selected].mean() if count else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def text_table(frame: pd.DataFrame) -> str:
    """Render a compact table without relying on the optional tabulate package."""
    return "```text\n" + frame.to_string() + "\n```"


def main() -> None:
    required_files = [
        METRICS_PATH,
        PROBABILITIES_PATH,
        ABLATION_PATH,
        REGIME_PATH,
        ANOMALY_PATH,
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run the existing Phase 5-8 scripts first; missing artifacts: "
            + ", ".join(missing)
        )

    metrics = pd.read_csv(METRICS_PATH, parse_dates=["validation_start", "validation_end"])
    probabilities = pd.read_csv(PROBABILITIES_PATH, parse_dates=["timestamp"])
    ablation = pd.read_csv(ABLATION_PATH)
    regimes = pd.read_csv(REGIME_PATH)
    anomalies = pd.read_csv(ANOMALY_PATH, parse_dates=["timestamp"])

    threshold_results = build_threshold_analysis(probabilities)
    threshold_path = OUTPUT_DIR / "confidence_threshold_analysis.csv"
    report_path = OUTPUT_DIR / "phase_9_research_summary.md"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    threshold_results.to_csv(threshold_path, index=False)

    metric_summary = metrics.groupby("model")[
        ["balanced_accuracy", "brier_score", "log_loss"]
    ].agg(["mean", "std"])

    # Pair each ablation fold with the all-features result from the same period.
    all_features = ablation[ablation["experiment"] == "all_features"][
        ["fold", "balanced_accuracy"]
    ].rename(columns={"balanced_accuracy": "all_features_balanced_accuracy"})
    ablation_delta = ablation.merge(all_features, on="fold", validate="many_to_one")
    ablation_delta["balanced_accuracy_change"] = (
        ablation_delta["balanced_accuracy"]
        - ablation_delta["all_features_balanced_accuracy"]
    )
    ablation_summary = ablation_delta.groupby("experiment")[
        "balanced_accuracy_change"
    ].agg(["mean", "std"])

    regime_summary = regimes.groupby("regime").agg(
        validation_rows_mean=("validation_rows", "mean"),
        up_rate_mean=("observed_up_rate", "mean"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
    )

    anomaly_summary = anomalies.groupby("is_anomaly").agg(
        observations=("timestamp", "size"),
        mean_abs_return=("return_1h", lambda values: values.abs().mean()),
        next_up_rate=("target_up", "mean"),
    )

    start = probabilities["timestamp"].min()
    end = probabilities["timestamp"].max()
    report = [
        "# Phase 9 research summary",
        "",
        f"Walk-forward out-of-fold validation period: {start} to {end}.",
        "The newest 15% final test period was excluded from these experiments.",
        "",
        "## Model comparison",
        "",
        text_table(metric_summary.round(4)),
        "",
        "Balanced accuracy and probability scores are averaged over chronological folds.",
        "",
        "## Feature ablation",
        "",
        text_table(ablation_summary.round(4)),
        "",
        "Positive change means removing that family scored above the all-feature model; "
        "negative change means it scored below. Fold-to-fold spread matters, so small "
        "differences are not proof that a feature family is useful or harmful.",
        "",
        "## Regime comparison",
        "",
        text_table(regime_summary.round(4)),
        "",
        "Regime sample sizes differ; results from small groups are uncertain.",
        "",
        "## Anomaly comparison",
        "",
        text_table(anomaly_summary.round(4)),
        "",
        "Anomaly flags describe unusual feature combinations, not bad data or trade signals.",
        "",
        "## Confidence threshold experiment",
        "",
        "See `confidence_threshold_analysis.csv`. Confidence is max(P(up), P(down)); "
        "coverage is the share of OOF validation rows meeting the threshold. Accuracy "
        "is measured only on covered rows. This is a research diagnostic, not a trading "
        "backtest or a guarantee of future accuracy.",
        "",
        "## Reproducibility note",
        "",
        "This report reads the saved walk-forward, ablation, regime, and anomaly outputs. "
        "The threshold experiment uses saved out-of-fold probabilities, so it does not "
        "refit models or inspect the reserved final test period.",
        "",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(f"OOF validation rows: {probabilities['timestamp'].nunique():,}")
    print(f"Validation period: {start} to {end}")
    print("Final test period: still reserved and untouched")
    print("\nConfidence threshold results (direction accuracy on covered rows):")
    print(threshold_results.round(2).to_string(index=False))
    print(f"\nSaved threshold analysis to: {threshold_path}")
    print(f"Saved combined research report to: {report_path}")


if __name__ == "__main__":
    main()
