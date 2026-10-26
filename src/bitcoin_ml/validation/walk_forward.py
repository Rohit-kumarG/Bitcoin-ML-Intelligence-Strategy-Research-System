"""Expanding-window evaluation on historical data before the final test period."""

from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


DATA_PATH = Path("data/processed/btcusdt_1h_features.csv")
OUTPUT_PATH = Path("artifacts/metrics/walk_forward_metrics.csv")
PROBABILITY_PATH = Path("artifacts/metrics/walk_forward_probabilities.csv")
CALIBRATION_PATH = Path("artifacts/metrics/calibration_bins.csv")
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
TARGET = "target_up"
N_SPLITS = 5
VALIDATION_HOURS = 24 * 182  # approximately six months per validation window
FINAL_TEST_FRACTION = 0.15


def calculate_metrics(actual: pd.Series, predicted, probability_up) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "brier_score": brier_score_loss(actual, probability_up),
        "log_loss": log_loss(actual, probability_up, labels=[0, 1]),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.dropna(subset=FEATURES + [TARGET]).sort_values("timestamp").reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)

    # Keep the newest 15% untouched for a later, final evaluation.
    development_size = int(len(df) * (1 - FINAL_TEST_FRACTION))
    development = df.iloc[:development_size].copy()
    final_test = df.iloc[development_size:]

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=VALIDATION_HOURS,
        gap=1,
    )

    fold_results = []
    probability_results = []
    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(development), start=1
    ):
        train = development.iloc[train_indices]
        validation = development.iloc[validation_indices]
        X_train, y_train = train[FEATURES], train[TARGET]
        X_validation, y_validation = validation[FEATURES], validation[TARGET]

        majority_class = y_train.mode().iloc[0]
        baseline_predictions = [majority_class] * len(validation)
        baseline_probability = y_train.mean()
        baseline_probabilities = [baseline_probability] * len(validation)

        logistic_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42),
        )
        random_forest = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )
        calibrated_random_forest = CalibratedClassifierCV(
            estimator=RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
            ),
            method="sigmoid",
            cv=TimeSeriesSplit(n_splits=3, gap=1),
            ensemble=True,
        )
        xgboost_model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        tuned_random_forest = GridSearchCV(
            estimator=RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),
            param_grid={
                "max_depth": [4, 8],
                "min_samples_leaf": [20, 50],
            },
            scoring="balanced_accuracy",
            cv=TimeSeriesSplit(n_splits=2, gap=1),
            refit=True,
            n_jobs=1,
        )

        models = [
            ("logistic_regression", logistic_model),
            ("random_forest", random_forest),
            ("random_forest_sigmoid_calibrated", calibrated_random_forest),
            ("random_forest_tuned", tuned_random_forest),
            ("xgboost", xgboost_model),
        ]
        model_predictions = []
        for model_name, model in models:
            model.fit(X_train, y_train)
            model_predictions.append(
                (
                    model_name,
                    model.predict(X_validation),
                    model.predict_proba(X_validation)[:, 1],
                    getattr(model, "best_params_", {}),
                )
            )

        predictions_to_score = [
            ("majority_baseline", baseline_predictions, baseline_probabilities, {}),
            *model_predictions,
        ]
        for model_name, predictions, probability_up, best_parameters in predictions_to_score:
            result = {
                "fold": fold_number,
                "model": model_name,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "train_start": train["timestamp"].min(),
                "train_end": train["timestamp"].max(),
                "validation_start": validation["timestamp"].min(),
                "validation_end": validation["timestamp"].max(),
                "best_parameters": str(best_parameters),
            }
            result.update(calculate_metrics(y_validation, predictions, probability_up))
            fold_results.append(result)

            probability_results.extend(
                {
                    "fold": fold_number,
                    "model": model_name,
                    "timestamp": timestamp,
                    "actual_up": int(actual),
                    "probability_up": float(probability),
                }
                for timestamp, actual, probability in zip(
                    validation["timestamp"], y_validation, probability_up
                )
            )

    results = pd.DataFrame(fold_results)
    probabilities = pd.DataFrame(probability_results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    probabilities.to_csv(PROBABILITY_PATH, index=False)

    calibration_rows = []
    for model_name, model_data in probabilities.groupby("model"):
        observed_frequency, mean_probability = calibration_curve(
            model_data["actual_up"],
            model_data["probability_up"],
            n_bins=10,
            strategy="quantile",
        )
        calibration_rows.extend(
            {
                "model": model_name,
                "bin": bin_number,
                "mean_predicted_probability": float(mean_prob),
                "observed_up_frequency": float(observed_freq),
            }
            for bin_number, (mean_prob, observed_freq) in enumerate(
                zip(mean_probability, observed_frequency), start=1
            )
        )
    pd.DataFrame(calibration_rows).to_csv(CALIBRATION_PATH, index=False)

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "brier_score",
        "log_loss",
    ]
    summary = results.groupby("model")[metric_names].agg(["mean", "std"])

    print(f"Usable rows: {len(df):,}")
    print(f"Development rows used in walk-forward folds: {len(development):,}")
    print(f"Final test rows reserved and not evaluated: {len(final_test):,}")
    print(f"Walk-forward folds: {N_SPLITS}; validation window: approximately six months each")
    print(
        "Models: majority baseline, Logistic Regression, Random Forest, "
        "sigmoid-calibrated Random Forest, tuned Random Forest, XGBoost"
    )
    print("\nFold date ranges:")
    print(
        results[results["model"] == "logistic_regression"][
            ["fold", "train_start", "train_end", "validation_start", "validation_end"]
        ].to_string(index=False)
    )
    print("\nMetric mean and standard deviation across folds:")
    print(summary.round(4).to_string())
    print("\nTuned Random Forest settings selected inside each training fold:")
    print(
        results.loc[results["model"] == "random_forest_tuned", ["fold", "best_parameters"]]
        .to_string(index=False)
    )
    print(f"\nSaved fold results to: {OUTPUT_PATH}")
    print(f"Saved out-of-fold probabilities to: {PROBABILITY_PATH}")
    print(f"Saved calibration-bin data to: {CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
