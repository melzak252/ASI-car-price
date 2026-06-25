from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from lightgbm import LGBMRegressor

from api.monitoring import check_drift, generate_prediction_id, log_prediction
from api.preprocessing import InferencePreprocessor
from api.schemas import CarAdFeatures, DriftReportResponse, HealthResponse, PredictionResponse

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / os.getenv("MODEL_PATH", "models/best_model.pkl")
PREPROC_PATH = ROOT / os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl")
SEL_FEAT_PATH = ROOT / os.getenv("SELECTED_FEATURES_PATH", "models/selected_feature_columns.joblib")

_model: LGBMRegressor | None = None
_preprocessor: InferencePreprocessor | None = None
_selected_features: list[str] | None = None
_model_loaded: bool = False


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _model, _preprocessor, _selected_features, _model_loaded

    if MODEL_PATH.exists():
        _model = joblib.load(str(MODEL_PATH))
        if PREPROC_PATH.exists():
            _preprocessor = InferencePreprocessor.load(str(PREPROC_PATH))
        if SEL_FEAT_PATH.exists():
            _selected_features = joblib.load(str(SEL_FEAT_PATH))
        _model_loaded = True
    yield


app = FastAPI(title="ASI Car Price API", lifespan=lifespan)


def _prepare_input(data: CarAdFeatures) -> pd.DataFrame:
    row = {
        "Price": data.price or np.nan, "Currency": data.currency or "PLN",
        "Condition": data.condition, "Vehicle_brand": data.vehicle_brand,
        "Vehicle_model": data.vehicle_model, "Vehicle_version": data.vehicle_version or np.nan,
        "Vehicle_generation": data.vehicle_generation or np.nan,
        "Production_year": data.production_year or np.nan, "Mileage_km": data.mileage_km or np.nan,
        "Power_HP": data.power_hp or np.nan, "Displacement_cm3": data.displacement_cm3 or np.nan,
        "Fuel_type": data.fuel_type, "CO2_emissions": data.co2_emissions or np.nan,
        "Drive": data.drive or "", "Transmission": data.transmission or np.nan,
        "Type": data.type, "Doors_number": data.doors_number or np.nan,
        "Colour": data.colour or np.nan, "Origin_country": data.origin_country or np.nan,
        "First_owner": data.first_owner or np.nan,
        "First_registration_date": data.first_registration_date or np.nan,
        "Offer_publication_date": data.offer_publication_date or np.nan,
        "Offer_location": data.offer_location or np.nan, "Features": str(data.features),
    }
    df = pd.DataFrame([row])
    from features.build_features import add_brand_model
    from preprocessing.cleaning import clean_features
    df = add_brand_model(df)
    df["feat_list"] = df["Features"].apply(clean_features)
    return df


def _predict(df: pd.DataFrame) -> tuple[float, float]:
    if _preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocessor not loaded")
    X = _preprocessor.transform(df)
    if _selected_features is not None:
        for col in _selected_features:
            if col not in X.columns:
                X[col] = 0.0
        X = X.reindex(columns=_selected_features, fill_value=0.0)
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    log_pred = float(_model.predict(X)[0])
    return float(np.exp(log_pred)), log_pred


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_model_loaded)


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CarAdFeatures) -> PredictionResponse:
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    pred_id = generate_prediction_id()
    price_pred, log_pred = _predict(_prepare_input(data))
    log_prediction(data.model_dump(), price_pred, log_pred, "best_model_lgbm_tuned", pred_id)
    return PredictionResponse(
        predicted_price_pln=round(price_pred, 2),
        predicted_log_price=round(log_pred, 6),
        model_version="best_model_lgbm_tuned", prediction_id=pred_id,
    )


@app.get("/drift", response_model=DriftReportResponse)
def drift_report() -> DriftReportResponse:
    if _preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocessor not loaded")
    return DriftReportResponse(**check_drift(_preprocessor))


@app.get("/predictions")
def list_predictions(limit: int = 20) -> list[dict]:
    log_path = ROOT / os.getenv("PREDICTIONS_LOG_PATH", "reports/predictions_log.csv")
    if not log_path.exists():
        return []
    return pd.read_csv(log_path).tail(limit).to_dict(orient="records")
