from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/processed/btcusdt_1h_features.csv")
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


def report_metrics(name, actual, predicted) -> None:
    print(f"\n{name}")
    print(f"Accuracy:          {accuracy_score(actual, predicted):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(actual, predicted):.4f}")
    print(f"Precision:         {precision_score(actual, predicted, zero_division=0):.4f}")
    print(f"Recall:            {recall_score(actual, predicted, zero_division=0):.4f}")
    print(f"F1 score:          {f1_score(actual, predicted, zero_division=0):.4f}")


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.dropna(subset=FEATURES + [TARGET]).sort_values("timestamp")
    df[TARGET] = df[TARGET].astype(int)

    # Oldest 70% for training, next 15% for validation, newest 15% reserved.
    n = len(df)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    # Purge one sample at each boundary to prevent adjacent target overlap.
    train = train.iloc[:-1]
    validation = validation.iloc[:-1]

    X_train, y_train = train[FEATURES], train[TARGET]
    X_validation, y_validation = validation[FEATURES], validation[TARGET]

    print(
        f"Training:   {len(train):,} rows, "
        f"{train['timestamp'].min()} to {train['timestamp'].max()}"
    )
    print(
        f"Validation: {len(validation):,} rows, "
        f"{validation['timestamp'].min()} to {validation['timestamp'].max()}"
    )
    print(f"Test reserved for final evaluation: {len(test):,} rows")

    # Benchmark: always predict the most common training class.
    majority_class = y_train.mode().iloc[0]
    baseline_predictions = [majority_class] * len(validation)
    report_metrics(
        "Majority-class baseline (validation)",
        y_validation,
        baseline_predictions,
    )

    # Scaler is fitted on training data only; then Logistic Regression is fitted.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(X_train, y_train)

    validation_predictions = model.predict(X_validation)
    report_metrics(
        "Logistic Regression (validation)",
        y_validation,
        validation_predictions,
    )

    print(
        "Confusion matrix "
        "[[actual 0 predicted 0, actual 0 predicted 1], "
        "[actual 1 predicted 0, actual 1 predicted 1]]:"
    )
    print(confusion_matrix(y_validation, validation_predictions, labels=[0, 1]))


if __name__ == "__main__":
    main()
