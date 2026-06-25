from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CarAdFeatures(BaseModel):
    vehicle_brand: str = Field(...)
    vehicle_model: str = Field("")
    vehicle_version: Optional[str] = Field(None)
    vehicle_generation: Optional[str] = Field(None)
    production_year: Optional[float] = Field(None)
    mileage_km: Optional[float] = Field(None)
    power_hp: Optional[float] = Field(None)
    displacement_cm3: Optional[float] = Field(None)
    fuel_type: str = Field(...)
    drive: str = Field("Front wheels")
    type: str = Field(...)
    doors_number: Optional[float] = Field(None)
    condition: str = Field("Used")
    features: list[str] = Field(default_factory=list)
    currency: str = Field("PLN")
    colour: Optional[str] = Field(None)
    origin_country: Optional[str] = Field(None)
    first_owner: Optional[float] = Field(None)
    co2_emissions: Optional[float] = Field(None)
    transmission: Optional[str] = Field(None)
    first_registration_date: Optional[str] = Field(None)
    offer_publication_date: Optional[str] = Field(None)
    offer_location: Optional[str] = Field(None)
    price: Optional[float] = Field(None)


class PredictionResponse(BaseModel):
    predicted_price_pln: float
    predicted_log_price: float
    model_version: str = "best_model_lgbm_tuned"
    prediction_id: str = ""


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False


class DriftReportResponse(BaseModel):
    drift_detected: bool = False
    n_features_compared: int = 0
    n_drifted_features: int = 0
    drifted_features: list[str] = Field(default_factory=list)
    message: str = ""
