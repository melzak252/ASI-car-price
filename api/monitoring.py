from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_LOG = ROOT / os.getenv("PREDICTIONS_LOG_PATH", "reports/predictions_log.csv")
REFERENCE_DATA = ROOT / os.getenv("REFERENCE_DATA_PATH", "data/05_model_input/X_train.pkl")
DRIFT_WINDOW = 100


def generate_prediction_id() -> str:
    return uuid.uuid4().hex[:12]


def log_prediction(raw_input: dict, predicted_price_pln: float, predicted_log_price: float, model_version: str, prediction_id: str) -> None:
    record = {
        "prediction_id": prediction_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predicted_price_pln": round(float(predicted_price_pln), 2),
        "predicted_log_price": round(float(predicted_log_price), 6),
        "model_version": model_version,
        "actual_price_pln": raw_input.get("price"),
        "vehicle_brand": raw_input.get("vehicle_brand", ""),
        "vehicle_model": raw_input.get("vehicle_model", ""),
        "production_year": raw_input.get("production_year"),
        "mileage_km": raw_input.get("mileage_km"),
        "power_hp": raw_input.get("power_hp"),
        "fuel_type": raw_input.get("fuel_type", ""),
        "type": raw_input.get("type", ""),
        "features_count": len(raw_input.get("features", [])),
    }
    pd.DataFrame([record]).to_csv(PREDICTIONS_LOG, mode="a", header=not PREDICTIONS_LOG.exists(), index=False)


def check_drift(preprocessor, reference_data=None, current_data=None) -> dict:
    if reference_data is None:
        ref = Path(REFERENCE_DATA)
        if not ref.exists():
            return {"drift_detected": False, "n_features_compared": 0, "n_drifted_features": 0, "drifted_features": [], "message": "Brak danych referencyjnych"}
        reference_data = pd.read_pickle(ref)

    if current_data is None:
        if not PREDICTIONS_LOG.exists():
            return {"drift_detected": False, "n_features_compared": 0, "n_drifted_features": 0, "drifted_features": [], "message": "Brak zapisanych predykcji"}
        log_df = pd.read_csv(PREDICTIONS_LOG)
        if len(log_df) < 10:
            return {"drift_detected": False, "n_features_compared": 0, "n_drifted_features": 0, "drifted_features": [], "message": f"Tylko {len(log_df)} predykcji, potrzeba 10+"}
        curr = log_df.tail(DRIFT_WINDOW)
        cols = ["production_year", "mileage_km", "power_hp"]
        ref_num = reference_data[[c for c in cols if c in reference_data.columns]]
        cur_num = curr[[c for c in cols if c in curr.columns]]
        if ref_num.empty or cur_num.empty:
            return {"drift_detected": False, "n_features_compared": 0, "n_drifted_features": 0, "drifted_features": [], "message": "Brak kolumn numerycznych"}
        drifted = []
        for col in ref_num.columns:
            m, s = ref_num[col].mean(), ref_num[col].std()
            if s == 0:
                continue
            if abs(cur_num[col].mean() - m) / (s / len(cur_num) ** 0.5) > 3.0:
                drifted.append(col)
        return {
            "drift_detected": len(drifted) > 0,
            "n_features_compared": len(ref_num.columns),
            "n_drifted_features": len(drifted),
            "drifted_features": drifted,
            "message": f"{len(drifted)}/{len(ref_num.columns)} cech dryfuje" if drifted else "Brak dryfu",
        }
    return {"drift_detected": False, "n_features_compared": 0, "n_drifted_features": 0, "drifted_features": [], "message": "Pominięto"}
