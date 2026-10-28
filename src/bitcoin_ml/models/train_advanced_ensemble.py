"""Train and save the advanced ensemble model for production inference.

Trains LightGBM + XGBoost + CatBoost + Random Forest on all available data,
saves each model artifact and a metadata JSON with feature list and weights.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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

from src.bitcoin_ml.features.build_advanced_features import ALL_ADVANCED_FEATURES


DATA_PATH = Path("data/processed/btcusdt_usdm_1h_advanced_features.csv")
MODEL_DIR = Path("artifacts/models")
METADATA_PATH = MODEL_DIR / "advanced_ensemble_v1.json"
TARGET = "target_up"
MODEL_VERSION = "advanced-ensemble-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_features(features: list[str], data: pd.DataFrame) -> list[str]:
    nan_frac = data[features].isna().mean()
    return nan_frac[nan_frac < 0.30].index.tolist()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Run run_advanced_feature_pipeline.py first: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    candidate_features = [f for f in ALL_ADVANCED_FEATURES if f in data.columns]
    data = data.dropna(subset=[TARGET]).sort_values("timestamp").reset_index(drop=True)
    data[TARGET] = data[TARGET].astype(int)

    features = select_features(candidate_features, data)
    train_medians = data[features].median()
    x = data[features].fillna(train_medians)
    y = data[TARGET]

    print(f"Training on {len(data):,} rows with {len(features)} features")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    saved_models = {}

    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=20,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    rf.fit(x, y)
    rf_path = MODEL_DIR / "advanced_random_forest_v1.joblib"
    joblib.dump(rf, rf_path)
    saved_models["random_forest"] = rf_path.name
    print(f"  Saved to {rf_path}")

    # LightGBM
    if HAS_LIGHTGBM:
        print("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=50, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        lgb_model.fit(x, y)
        lgb_path = MODEL_DIR / "advanced_lightgbm_v1.joblib"
        joblib.dump(lgb_model, lgb_path)
        saved_models["lightgbm"] = lgb_path.name
        print(f"  Saved to {lgb_path}")

    # XGBoost
    if HAS_XGBOOST:
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            n_jobs=-1, eval_metric="logloss", verbosity=0,
        )
        xgb_model.fit(x, y)
        xgb_path = MODEL_DIR / "advanced_xgboost_v1.joblib"
        joblib.dump(xgb_model, xgb_path)
        saved_models["xgboost"] = xgb_path.name
        print(f"  Saved to {xgb_path}")

    # CatBoost
    if HAS_CATBOOST:
        print("Training CatBoost...")
        cb_model = cb.CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, random_seed=42, verbose=0, thread_count=-1,
        )
        cb_model.fit(x, y)
        cb_path = MODEL_DIR / "advanced_catboost_v1.joblib"
        joblib.dump(cb_model, cb_path)
        saved_models["catboost"] = cb_path.name
        print(f"  Saved to {cb_path}")

    # Save medians for inference-time NaN filling.
    medians_path = MODEL_DIR / "advanced_feature_medians_v1.joblib"
    joblib.dump(train_medians.to_dict(), medians_path)

    metadata = {
        "model_version": MODEL_VERSION,
        "models": saved_models,
        "ensemble_method": "auc_weighted_soft_vote",
        "features": features,
        "feature_medians_path": medians_path.name,
        "target": TARGET,
        "positive_class_meaning": "The next consecutive 1-hour Futures close return is positive.",
        "training_rows": len(data),
        "training_start": data["timestamp"].min().isoformat(),
        "training_end": data["timestamp"].max().isoformat(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_data_path": DATA_PATH.as_posix(),
        "training_data_sha256": sha256_file(DATA_PATH),
        "usage": "Research estimate only; not a guarantee or an order instruction.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nModel version: {MODEL_VERSION}")
    print(f"Models saved: {list(saved_models.keys())}")
    print(f"Training rows: {len(data):,}")
    print(f"Features: {len(features)}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
