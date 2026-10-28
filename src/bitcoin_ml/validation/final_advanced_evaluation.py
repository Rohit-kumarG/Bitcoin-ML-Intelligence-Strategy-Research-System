"""One-time final holdout evaluation for the advanced ensemble model.

Trains the full ensemble on the development set, evaluates on the reserved
final 15% test period, and produces comprehensive metrics including
per-model performance, ensemble accuracy, and high-confidence filtered results.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from src.bitcoin_ml.features.build_advanced_features import ALL_ADVANCED_FEATURES


DATA_PATH = Path("data/processed/btcusdt_usdm_1h_advanced_features.csv")
FUNDING_PATH = Path("data/raw/btcusdt_usdm_funding.csv")
PREDICTIONS_PATH = Path("artifacts/metrics/advanced_final_test_predictions.csv")
MODEL_METRICS_PATH = Path("artifacts/metrics/advanced_final_test_model_metrics.csv")
CONFIDENCE_PATH = Path("artifacts/metrics/advanced_final_test_confidence.csv")
STRATEGY_PATH = Path("artifacts/metrics/advanced_final_test_strategy_grid.csv")
TARGET = "target_up"
FINAL_TEST_FRACTION = 0.15
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75]
COST_SCENARIOS = [(0.001, 0.0005), (0.001, 0.001), (0.002, 0.002)]
INITIAL_CAPITAL = 10_000.0


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def select_features(features: list[str], data: pd.DataFrame) -> list[str]:
    nan_frac = data[features].isna().mean()
    return nan_frac[nan_frac < 0.30].index.tolist()


def main() -> None:
    for path in [DATA_PATH, FUNDING_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")

    data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    candidate_features = [f for f in ALL_ADVANCED_FEATURES if f in data.columns]
    data = data.dropna(subset=[TARGET, "next_return"]).sort_values("timestamp").reset_index(drop=True)
    data[TARGET] = data[TARGET].astype(int)

    test_start = int(len(data) * (1 - FINAL_TEST_FRACTION))
    train = data.iloc[:test_start - 1]
    test = data.iloc[test_start:].copy()

    features = select_features(candidate_features, train)
    print(f"Features used: {len(features)}")
    print(f"Training rows: {len(train):,}")
    print(f"Final test rows: {len(test):,}")
    print(f"Test period: {test['timestamp'].min()} to {test['timestamp'].max()}")

    train_medians = train[features].median()
    x_train = train[features].fillna(train_medians)
    x_test = test[features].fillna(train_medians)
    y_train = train[TARGET]
    y_test = test[TARGET]

    model_probs = {}
    model_metrics = []

    # ---- Random Forest ----
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=20,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    rf.fit(x_train, y_train)
    model_probs["random_forest"] = rf.predict_proba(x_test)[:, 1]

    # ---- LightGBM ----
    if HAS_LIGHTGBM:
        print("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=50, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        lgb_model.fit(x_train, y_train)
        model_probs["lightgbm"] = lgb_model.predict_proba(x_test)[:, 1]

    # ---- XGBoost ----
    if HAS_XGBOOST:
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            n_jobs=-1, eval_metric="logloss", verbosity=0,
        )
        xgb_model.fit(x_train, y_train)
        model_probs["xgboost"] = xgb_model.predict_proba(x_test)[:, 1]

    # ---- CatBoost ----
    if HAS_CATBOOST:
        print("Training CatBoost...")
        cb_model = cb.CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, random_seed=42, verbose=0, thread_count=-1,
        )
        cb_model.fit(x_train, y_train)
        model_probs["catboost"] = cb_model.predict_proba(x_test)[:, 1]

    # ---- Per-model metrics ----
    for name, prob in model_probs.items():
        pred = (prob >= 0.5).astype(int)
        m = {
            "model": name,
            "train_rows": len(train),
            "test_rows": len(test),
            "test_start": test["timestamp"].min(),
            "test_end": test["timestamp"].max(),
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob),
            "brier_score": brier_score_loss(y_test, prob),
            "log_loss": log_loss(y_test, prob, labels=[0, 1]),
        }
        model_metrics.append(m)
        print(f"\n{name.upper()}: accuracy={m['accuracy']:.4f} "
              f"bal_acc={m['balanced_accuracy']:.4f} auc={m['roc_auc']:.4f}")

    # ---- Ensemble ----
    # AUC-weighted ensemble.
    auc_weights = {m["model"]: m["roc_auc"] for m in model_metrics}
    total_w = sum(auc_weights.values())
    ensemble_prob = sum(
        auc_weights[name] / total_w * prob
        for name, prob in model_probs.items()
    )
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)
    ensemble_m = {
        "model": "ensemble",
        "train_rows": len(train),
        "test_rows": len(test),
        "test_start": test["timestamp"].min(),
        "test_end": test["timestamp"].max(),
        "accuracy": accuracy_score(y_test, ensemble_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, ensemble_pred),
        "precision": precision_score(y_test, ensemble_pred, zero_division=0),
        "recall": recall_score(y_test, ensemble_pred, zero_division=0),
        "f1": f1_score(y_test, ensemble_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, ensemble_prob),
        "brier_score": brier_score_loss(y_test, ensemble_prob),
        "log_loss": log_loss(y_test, ensemble_prob, labels=[0, 1]),
    }
    model_metrics.append(ensemble_m)

    print(f"\n{'='*60}")
    print(f"ENSEMBLE: accuracy={ensemble_m['accuracy']:.4f} "
          f"bal_acc={ensemble_m['balanced_accuracy']:.4f} "
          f"auc={ensemble_m['roc_auc']:.4f}")
    print(f"{'='*60}")

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_test, ensemble_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{classification_report(y_test, ensemble_pred, target_names=['DOWN', 'UP'])}")

    # ---- High-confidence analysis ----
    confidence_rows = []
    for threshold in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        confident = test[
            (ensemble_prob >= threshold) | (ensemble_prob <= (1 - threshold))
        ]
        conf_prob = ensemble_prob[confident.index - test.index[0]]
        if len(confident) == 0:
            continue
        conf_pred = (conf_prob >= 0.5).astype(int)
        confidence_rows.append({
            "threshold": threshold,
            "covered_rows": len(confident),
            "coverage_pct": 100 * len(confident) / len(test),
            "accuracy": accuracy_score(confident[TARGET], conf_pred),
            "balanced_accuracy": balanced_accuracy_score(confident[TARGET], conf_pred),
        })

    confidence_df = pd.DataFrame(confidence_rows)

    print(f"\nHIGH-CONFIDENCE FILTERING (Final Test):")
    print(confidence_df.round(4).to_string(index=False))

    # ---- Strategy grid with funding ----
    funding = pd.read_csv(FUNDING_PATH, parse_dates=["timestamp"])
    funding["timestamp"] = funding["timestamp"] - pd.Timedelta(hours=2)
    test = test.merge(
        funding.rename(columns={"funding_rate": "settlement_funding_rate"})[
            ["timestamp", "settlement_funding_rate"]
        ],
        on="timestamp",
        how="left",
        validate="one_to_one",
    )
    test["settlement_funding_rate"] = test["settlement_funding_rate"].fillna(0.0)

    elapsed_years = max(
        (test["timestamp"].iloc[-1] - test["timestamp"].iloc[0]).total_seconds()
        / (365.25 * 24 * 60 * 60),
        1 / (365.25 * 24),
    )
    summaries = []
    for fee_per_side, slippage_per_side in COST_SCENARIOS:
        cost = fee_per_side + slippage_per_side
        for threshold in THRESHOLDS:
            run = test.copy()
            run["position"] = 0
            run.loc[ensemble_prob >= threshold, "position"] = 1
            run.loc[ensemble_prob <= 1 - threshold, "position"] = -1
            previous = run["position"].shift(1, fill_value=0)
            change_units = (run["position"] - previous).abs()
            pnl = run["position"] * run["next_return"]
            funding_pnl = -run["position"] * run["settlement_funding_rate"]
            net_return = (1 + pnl + funding_pnl) * (1 - cost) ** change_units - 1
            final_exit_units = abs(int(run["position"].iloc[-1]))
            if final_exit_units:
                net_return.iloc[-1] = (1 + net_return.iloc[-1]) * (1 - cost) ** final_exit_units - 1
            equity = INITIAL_CAPITAL * (1 + net_return).cumprod()
            active = run["position"] != 0
            cagr = (equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / elapsed_years) - 1
            summaries.append({
                "threshold": threshold,
                "fee_per_side_pct": 100 * fee_per_side,
                "slippage_per_side_pct": 100 * slippage_per_side,
                "long_hours": int((run["position"] == 1).sum()),
                "short_hours": int((run["position"] == -1).sum()),
                "time_in_market_pct": 100 * active.mean(),
                "ending_equity": equity.iloc[-1],
                "total_return_pct": 100 * (equity.iloc[-1] / INITIAL_CAPITAL - 1),
                "cagr_pct": 100 * cagr,
                "max_drawdown_pct": 100 * max_drawdown(equity),
            })

    # Save everything.
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    test_output = test[["timestamp", TARGET, "next_return", "settlement_funding_rate"]].copy()
    test_output["ensemble_probability_up"] = ensemble_prob
    test_output["ensemble_predicted_up"] = ensemble_pred
    for name, prob in model_probs.items():
        test_output[f"prob_{name}"] = prob
    test_output.to_csv(PREDICTIONS_PATH, index=False)

    pd.DataFrame(model_metrics).to_csv(MODEL_METRICS_PATH, index=False)
    confidence_df.to_csv(CONFIDENCE_PATH, index=False)
    pd.DataFrame(summaries).to_csv(STRATEGY_PATH, index=False)

    print(f"\nSaved final test predictions to: {PREDICTIONS_PATH}")
    print(f"Saved model metrics to: {MODEL_METRICS_PATH}")
    print(f"Saved confidence analysis to: {CONFIDENCE_PATH}")
    print(f"Saved strategy grid to: {STRATEGY_PATH}")
    print("\nONE-TIME evaluation; do not tune from these results.")


if __name__ == "__main__":
    main()
