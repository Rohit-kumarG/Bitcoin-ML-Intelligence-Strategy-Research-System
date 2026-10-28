"""Train and save the frozen Futures Random Forest for later API inference."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = Path("data/processed/btcusdt_usdm_1h_features.csv")
MODEL_PATH = Path("artifacts/models/futures_random_forest_v1.joblib")
METADATA_PATH = Path("artifacts/models/futures_random_forest_v1.json")
FEATURES = [
    "return_1h", "return_3h", "return_6h", "return_24h", "volatility_24h",
    "volume_log_change_1h", "candle_body_pct", "candle_range_pct",
    "upper_wick_pct", "lower_wick_pct",
]
TARGET = "target_up"
MODEL_VERSION = "futures-rf-v1"
MODEL_PARAMETERS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "random_state": 42,
    "n_jobs": -1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Run the Futures feature pipeline first: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    data = data.dropna(subset=FEATURES + [TARGET]).sort_values("timestamp").reset_index(drop=True)
    data[TARGET] = data[TARGET].astype(int)

    model = RandomForestClassifier(**MODEL_PARAMETERS)
    model.fit(data[FEATURES], data[TARGET])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "sklearn.ensemble.RandomForestClassifier",
        "target": TARGET,
        "positive_class_meaning": "The next consecutive 1-hour Futures close return is positive.",
        "features": FEATURES,
        "parameters": MODEL_PARAMETERS,
        "training_rows": len(data),
        "training_start": data["timestamp"].min().isoformat(),
        "training_end": data["timestamp"].max().isoformat(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_data_path": DATA_PATH.as_posix(),
        "training_data_sha256": sha256_file(DATA_PATH),
        "final_holdout_note": (
            "The one-time final holdout evaluation was recorded before this production fit. "
            "The fixed model settings were not tuned from that result; all currently labeled "
            "rows are included for deployment training, so the historical holdout is no longer "
            "untouched for future experiments."
        ),
        "usage": "Research estimate only; not a guarantee or an order instruction.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Model version: {MODEL_VERSION}")
    print(f"Training rows: {len(data):,}")
    print(f"Training range: {metadata['training_start']} to {metadata['training_end']}")
    print("Features:", ", ".join(FEATURES))
    print(f"Saved trained model to: {MODEL_PATH}")
    print(f"Saved model metadata to: {METADATA_PATH}")
    print("No training accuracy is reported as an estimate of future performance.")


if __name__ == "__main__":
    main()
