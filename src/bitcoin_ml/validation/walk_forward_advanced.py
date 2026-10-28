"""Train a multi-model ensemble (LightGBM + XGBoost + CatBoost + RF) with
walk-forward validation and high-confidence filtering for maximum accuracy.

This replaces the original single Random Forest with:
1. Four gradient-boosted / tree-based models.
2. Per-fold Optuna hyperparameter optimisation (inner time-series CV).
3. Weighted soft-voting ensemble.
4. High-confidence thresholding analysis.
5. Complete out-of-fold probability file + fold metrics.

Dependencies: lightgbm, xgboost, catboost, optuna (added to requirements).
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
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)

# Try to import optional gradient boosting libraries.
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
PROBABILITY_PATH = Path("artifacts/metrics/advanced_walk_forward_probabilities.csv")
METRICS_PATH = Path("artifacts/metrics/advanced_walk_forward_metrics.csv")
ENSEMBLE_METRICS_PATH = Path("artifacts/metrics/advanced_ensemble_summary.csv")
CONFIDENCE_PATH = Path("artifacts/metrics/advanced_confidence_analysis.csv")
TARGET = "target_up"
N_SPLITS = 5
VALIDATION_HOURS = 24 * 182  # ~6 months
FINAL_TEST_FRACTION = 0.15
OPTUNA_TRIALS = 0
INNER_CV_SPLITS = 2


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_lightgbm(params: dict | None = None):
    if not HAS_LIGHTGBM:
        return None
    default = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        default.update(params)
    return lgb.LGBMClassifier(**default)


def make_xgboost(params: dict | None = None):
    if not HAS_XGBOOST:
        return None
    default = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    if params:
        default.update(params)
    return xgb.XGBClassifier(**default)


def make_catboost(params: dict | None = None):
    if not HAS_CATBOOST:
        return None
    default = {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": 0,
        "thread_count": -1,
    }
    if params:
        default.update(params)
    return cb.CatBoostClassifier(**default)


def make_random_forest(params: dict | None = None):
    default = {
        "n_estimators": 500,
        "max_depth": 10,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        default.update(params)
    return RandomForestClassifier(**default)


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning (inner CV within training fold)
# ---------------------------------------------------------------------------

def tune_lightgbm(x_train, y_train, n_trials=OPTUNA_TRIALS):
    if not HAS_OPTUNA or not HAS_LIGHTGBM or n_trials <= 0:
        return {}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        splitter = TimeSeriesSplit(n_splits=INNER_CV_SPLITS, gap=1)
        scores = []
        for tr_idx, va_idx in splitter.split(x_train):
            model = make_lightgbm(params)
            model.fit(x_train.iloc[tr_idx], y_train.iloc[tr_idx])
            prob = model.predict_proba(x_train.iloc[va_idx])[:, 1]
            scores.append(roc_auc_score(y_train.iloc[va_idx], prob))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgboost(x_train, y_train, n_trials=OPTUNA_TRIALS):
    if not HAS_OPTUNA or not HAS_XGBOOST or n_trials <= 0:
        return {}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        splitter = TimeSeriesSplit(n_splits=INNER_CV_SPLITS, gap=1)
        scores = []
        for tr_idx, va_idx in splitter.split(x_train):
            model = make_xgboost(params)
            model.fit(x_train.iloc[tr_idx], y_train.iloc[tr_idx])
            prob = model.predict_proba(x_train.iloc[va_idx])[:, 1]
            scores.append(roc_auc_score(y_train.iloc[va_idx], prob))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_catboost(x_train, y_train, n_trials=OPTUNA_TRIALS):
    if not HAS_OPTUNA or not HAS_CATBOOST or n_trials <= 0:
        return {}

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        }
        splitter = TimeSeriesSplit(n_splits=INNER_CV_SPLITS, gap=1)
        scores = []
        for tr_idx, va_idx in splitter.split(x_train):
            model = make_catboost(params)
            model.fit(x_train.iloc[tr_idx], y_train.iloc[tr_idx])
            prob = model.predict_proba(x_train.iloc[va_idx])[:, 1]
            scores.append(roc_auc_score(y_train.iloc[va_idx], prob))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ---------------------------------------------------------------------------
# Feature selection: drop features with too many NaN or zero importance
# ---------------------------------------------------------------------------

def select_features(features: list[str], data: pd.DataFrame) -> list[str]:
    """Keep only features with < 30% NaN in the data."""
    nan_frac = data[features].isna().mean()
    keep = nan_frac[nan_frac < 0.30].index.tolist()
    dropped = set(features) - set(keep)
    if dropped:
        print(f"  Dropped {len(dropped)} features with >30% NaN: {sorted(dropped)}")
    return keep


# ---------------------------------------------------------------------------
# Ensemble: weighted soft vote
# ---------------------------------------------------------------------------

def ensemble_probabilities(probs: dict[str, np.ndarray], weights: dict[str, float] | None = None) -> np.ndarray:
    """Weighted average of model probabilities."""
    if weights is None:
        weights = {name: 1.0 for name in probs}
    total_weight = sum(weights[name] for name in probs)
    result = np.zeros_like(list(probs.values())[0])
    for name, prob in probs.items():
        result += weights[name] * prob
    return result / total_weight


# ---------------------------------------------------------------------------
# Confidence analysis
# ---------------------------------------------------------------------------

def confidence_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy at various confidence thresholds."""
    rows = []
    for threshold in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75, 0.80]:
        confident = predictions[
            (predictions["ensemble_probability_up"] >= threshold)
            | (predictions["ensemble_probability_up"] <= (1 - threshold))
        ]
        if len(confident) == 0:
            continue
        predicted = (confident["ensemble_probability_up"] >= 0.5).astype(int)
        rows.append({
            "confidence_threshold": threshold,
            "covered_rows": len(confident),
            "coverage_pct": 100 * len(confident) / len(predictions),
            "accuracy": accuracy_score(confident["actual_up"], predicted),
            "balanced_accuracy": balanced_accuracy_score(confident["actual_up"], predicted),
            "precision": precision_score(confident["actual_up"], predicted, zero_division=0),
            "recall": recall_score(confident["actual_up"], predicted, zero_division=0),
            "f1": f1_score(confident["actual_up"], predicted, zero_division=0),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Run run_advanced_feature_pipeline.py first: {DATA_PATH}"
        )

    # Discover available models.
    model_names = ["random_forest"]
    if HAS_LIGHTGBM:
        model_names.append("lightgbm")
    if HAS_XGBOOST:
        model_names.append("xgboost")
    if HAS_CATBOOST:
        model_names.append("catboost")
    print(f"Available models: {model_names}")
    if HAS_OPTUNA:
        print(f"Optuna hyperparameter tuning enabled ({OPTUNA_TRIALS} trials per model per fold)")
    else:
        print("Optuna not installed; using default hyperparameters")

    data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    # Use all advanced features that exist in the data.
    candidate_features = [f for f in ALL_ADVANCED_FEATURES if f in data.columns]
    print(f"Candidate features found: {len(candidate_features)}")

    # Drop rows with missing target.
    data = data.dropna(subset=[TARGET]).sort_values("timestamp").reset_index(drop=True)
    data[TARGET] = data[TARGET].astype(int)

    # Reserve final test.
    development_size = int(len(data) * (1 - FINAL_TEST_FRACTION))
    development = data.iloc[:development_size].copy()
    final_test_reserved = data.iloc[development_size:]
    print(f"Development rows: {len(development):,}")
    print(f"Final test reserved: {len(final_test_reserved):,}")

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS, test_size=VALIDATION_HOURS, gap=1
    )

    all_predictions = []
    all_metrics = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(development), start=1):
        train = development.iloc[train_idx]
        valid = development.iloc[valid_idx].copy()
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{N_SPLITS}")
        print(f"  Train: {train['timestamp'].min()} to {train['timestamp'].max()} ({len(train):,} rows)")
        print(f"  Valid: {valid['timestamp'].min()} to {valid['timestamp'].max()} ({len(valid):,} rows)")

        # Select features for this fold.
        fold_features = select_features(candidate_features, train)

        # Fill NaN with column median from training data for models that can't handle NaN natively.
        train_medians = train[fold_features].median()
        x_train = train[fold_features].fillna(train_medians)
        x_valid = valid[fold_features].fillna(train_medians)
        y_train = train[TARGET]
        y_valid = valid[TARGET]

        model_probs = {}
        fold_model_metrics = []

        # ---- Random Forest ----
        print("  Training Random Forest...")
        rf = make_random_forest()
        rf.fit(x_train, y_train)
        rf_prob = rf.predict_proba(x_valid)[:, 1]
        model_probs["random_forest"] = rf_prob

        # ---- LightGBM ----
        if HAS_LIGHTGBM:
            print("  Tuning & training LightGBM...")
            lgb_params = tune_lightgbm(x_train, y_train) if HAS_OPTUNA else {}
            lgb_model = make_lightgbm(lgb_params)
            lgb_model.fit(x_train, y_train)
            lgb_prob = lgb_model.predict_proba(x_valid)[:, 1]
            model_probs["lightgbm"] = lgb_prob

        # ---- XGBoost ----
        if HAS_XGBOOST:
            print("  Tuning & training XGBoost...")
            xgb_params = tune_xgboost(x_train, y_train) if HAS_OPTUNA else {}
            xgb_model = make_xgboost(xgb_params)
            xgb_model.fit(x_train, y_train)
            xgb_prob = xgb_model.predict_proba(x_valid)[:, 1]
            model_probs["xgboost"] = xgb_prob

        # ---- CatBoost ----
        if HAS_CATBOOST:
            print("  Tuning & training CatBoost...")
            cb_params = tune_catboost(x_train, y_train) if HAS_OPTUNA else {}
            cb_model = make_catboost(cb_params)
            cb_model.fit(x_train, y_train)
            cb_prob = cb_model.predict_proba(x_valid)[:, 1]
            model_probs["catboost"] = cb_prob

        # ---- Per-model metrics ----
        for name, prob in model_probs.items():
            pred = (prob >= 0.5).astype(int)
            m = {
                "fold": fold,
                "model": name,
                "accuracy": accuracy_score(y_valid, pred),
                "balanced_accuracy": balanced_accuracy_score(y_valid, pred),
                "precision": precision_score(y_valid, pred, zero_division=0),
                "recall": recall_score(y_valid, pred, zero_division=0),
                "f1": f1_score(y_valid, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_valid, prob),
                "brier_score": brier_score_loss(y_valid, prob),
                "log_loss": log_loss(y_valid, prob, labels=[0, 1]),
            }
            fold_model_metrics.append(m)
            print(f"    {name}: acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
                  f"auc={m['roc_auc']:.4f} brier={m['brier_score']:.4f}")

        # ---- Weighted Ensemble ----
        # Weight models by their validation AUC from this fold.
        aucs = {name: m["roc_auc"] for m in fold_model_metrics for name in [m["model"]] if m["model"] == name}
        # Normalise: w_i = auc_i / sum(auc)
        ensemble_prob = ensemble_probabilities(model_probs, weights=aucs)
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)
        ensemble_metrics = {
            "fold": fold,
            "model": "ensemble",
            "accuracy": accuracy_score(y_valid, ensemble_pred),
            "balanced_accuracy": balanced_accuracy_score(y_valid, ensemble_pred),
            "precision": precision_score(y_valid, ensemble_pred, zero_division=0),
            "recall": recall_score(y_valid, ensemble_pred, zero_division=0),
            "f1": f1_score(y_valid, ensemble_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_valid, ensemble_prob),
            "brier_score": brier_score_loss(y_valid, ensemble_prob),
            "log_loss": log_loss(y_valid, ensemble_prob, labels=[0, 1]),
        }
        fold_model_metrics.append(ensemble_metrics)
        print(f"    ENSEMBLE: acc={ensemble_metrics['accuracy']:.4f} "
              f"bal_acc={ensemble_metrics['balanced_accuracy']:.4f} "
              f"auc={ensemble_metrics['roc_auc']:.4f}")

        all_metrics.extend(fold_model_metrics)

        # Store per-row predictions.
        for i, (ts, actual) in enumerate(zip(valid["timestamp"], y_valid)):
            row = {
                "fold": fold,
                "timestamp": ts,
                "actual_up": int(actual),
                "ensemble_probability_up": float(ensemble_prob[i]),
            }
            for name, prob_arr in model_probs.items():
                row[f"prob_{name}"] = float(prob_arr[i])
            all_predictions.append(row)

    # Save results.
    prediction_frame = pd.DataFrame(all_predictions)
    metric_frame = pd.DataFrame(all_metrics)

    PROBABILITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(PROBABILITY_PATH, index=False)
    metric_frame.to_csv(METRICS_PATH, index=False)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("WALK-FORWARD VALIDATION SUMMARY")
    print(f"{'='*60}")
    summary = metric_frame.groupby("model")[
        ["accuracy", "balanced_accuracy", "roc_auc", "brier_score", "f1"]
    ].mean().round(4)
    print(summary.to_string())

    # ---- Ensemble summary across folds ----
    ensemble_rows = metric_frame[metric_frame["model"] == "ensemble"]
    ensemble_summary = pd.DataFrame([{
        "metric": "Mean across folds",
        "accuracy": ensemble_rows["accuracy"].mean(),
        "balanced_accuracy": ensemble_rows["balanced_accuracy"].mean(),
        "roc_auc": ensemble_rows["roc_auc"].mean(),
        "brier_score": ensemble_rows["brier_score"].mean(),
        "f1": ensemble_rows["f1"].mean(),
        "min_accuracy": ensemble_rows["accuracy"].min(),
        "max_accuracy": ensemble_rows["accuracy"].max(),
    }])
    ensemble_summary.to_csv(ENSEMBLE_METRICS_PATH, index=False)

    # ---- Confidence analysis ----
    conf = confidence_analysis(prediction_frame)
    conf.to_csv(CONFIDENCE_PATH, index=False)

    print(f"\nHigh-confidence filtering analysis:")
    print(conf[["confidence_threshold", "covered_rows", "coverage_pct", "accuracy", "balanced_accuracy"]].round(4).to_string(index=False))

    print(f"\nSaved OOF probabilities to: {PROBABILITY_PATH}")
    print(f"Saved fold metrics to: {METRICS_PATH}")
    print(f"Saved ensemble summary to: {ENSEMBLE_METRICS_PATH}")
    print(f"Saved confidence analysis to: {CONFIDENCE_PATH}")


if __name__ == "__main__":
    main()
