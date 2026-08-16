"""
api.py
FastAPI service exposing the trained model as a POST /predict endpoint.

Run with:  uvicorn app.api:app --reload   (from the project root)

Kept deliberately simple per project constraints (section 15/20): one
endpoint, no auth/versioning layer, since over-engineering this would take
time away from the core ML work.
"""

import os
import sys
import pickle
import json

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import BUSINESS_ASSUMPTIONS  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

app = FastAPI(title="Demand Forecasting & Inventory API", version="1.0")

_model = None
_feature_cols = None


def _load_model():
    global _model, _feature_cols
    if _model is None:
        with open(os.path.join(MODELS_DIR, "sklearn_gbm_model.pkl"), "rb") as f:
            _model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "feature_cols.json")) as f:
            _feature_cols = json.load(f)
    return _model, _feature_cols


class PredictRequest(BaseModel):
    store_id: str
    product_id: str
    date: str                      # ISO date, e.g. "2024-01-15"
    recent_sales_lag_1: float
    recent_sales_lag_7: float
    recent_sales_lag_14: float
    recent_sales_lag_28: float
    rolling_mean_7: float
    rolling_mean_14: float
    rolling_mean_28: float
    rolling_std_7: float
    rolling_std_28: float
    inventory_level: float
    promotion: int
    price: float
    discount_pct: float
    competitor_price: float
    category_expanding_mean: float
    weather: str = "Sunny"
    category: str = "Groceries"
    region: str = "North"


class PredictResponse(BaseModel):
    predicted_demand: float
    stockout_risk: str
    recommended_order_qty: float


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model, feature_cols = _load_model()

    dt = pd.to_datetime(req.date)
    row = {
        "lag_1": req.recent_sales_lag_1,
        "lag_7": req.recent_sales_lag_7,
        "lag_14": req.recent_sales_lag_14,
        "lag_28": req.recent_sales_lag_28,
        "rolling_mean_7": req.rolling_mean_7,
        "rolling_mean_14": req.rolling_mean_14,
        "rolling_mean_28": req.rolling_mean_28,
        "rolling_std_7": req.rolling_std_7,
        "rolling_std_28": req.rolling_std_28,
        "day_of_week": dt.dayofweek,
        "week_of_year": int(dt.isocalendar()[1]),
        "month": dt.month,
        "quarter": dt.quarter,
        "year": dt.year,
        "is_weekend": int(dt.dayofweek >= 5),
        "promotion": req.promotion,
        "price": req.price,
        "inventory_level": req.inventory_level,
        "discount_pct": req.discount_pct,
        "price_vs_competitor": req.price - req.competitor_price,
        "price_vs_competitor_pct": (req.price - req.competitor_price) / req.competitor_price if req.competitor_price else 0,
        "cat_expanding_mean": req.category_expanding_mean,
        # Categorical encodings are simplified here (hash-based) since the
        # full train-fit category->code mapping isn't shipped with this demo
        # endpoint; production deployment should load and reuse the exact
        # mapping saved during training (see src/forecasting.py::_encode_categoricals).
        "store_id_enc": abs(hash(req.store_id)) % 5,
        "product_id_enc": abs(hash(req.product_id)) % 20,
        "category_enc": abs(hash(req.category)) % 5,
        "region_enc": abs(hash(req.region)) % 4,
        "weather_enc": abs(hash(req.weather)) % 4,
    }

    X = pd.DataFrame([row])[feature_cols]
    pred = float(np.clip(model.predict(X)[0], 0, None))

    lead_time = BUSINESS_ASSUMPTIONS["lead_time_days"]
    z = BUSINESS_ASSUMPTIONS["service_level_z"]
    sigma = max(req.rolling_std_7, 1.0)
    safety_stock = z * sigma * (lead_time ** 0.5)
    reorder_point = pred * lead_time + safety_stock
    recommended_order = max(0.0, reorder_point - req.inventory_level)

    if req.inventory_level <= safety_stock:
        risk = "🔴 Stockout Risk"
    elif req.inventory_level <= reorder_point:
        risk = "🟡 Low Inventory"
    elif req.inventory_level <= reorder_point * 2.5:
        risk = "🟢 Healthy Inventory"
    else:
        risk = "🔵 Overstock"

    return PredictResponse(
        predicted_demand=round(pred, 1),
        stockout_risk=risk,
        recommended_order_qty=round(recommended_order, 0),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
