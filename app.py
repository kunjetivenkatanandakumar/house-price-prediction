"""
app.py  —  House Price Prediction API
======================================
Endpoints:
  GET  /              -> health check
  GET  /docs          -> Swagger UI (auto-generated)
  POST /predict       -> predict from JSON input
  POST /predict/csv   -> predict from uploaded CSV file

Run locally:
  uvicorn app:app --reload --port 8000

Deploy to Render:
  Build command : pip install -r requirements.txt
  Start command : uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import io

app = FastAPI(
    title="House Price Predictor",
    description="Predict Ames, Iowa house prices using an ensemble ML model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
models = {}
feature_names = []
category_maps = {}
params = {}

@app.on_event("startup")
def load_models():
    global models, feature_names, category_maps, params
    models = {
        name: joblib.load(os.path.join(MODELS_DIR, f"{name}_model.pkl"))
        for name in ["ridge", "lasso", "rf", "gbm"]
    }
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    category_maps = joblib.load(os.path.join(MODELS_DIR, "category_maps.pkl"))
    with open(os.path.join(MODELS_DIR, "preprocessing_params.json")) as f:
        params = json.load(f)
    print(f"Loaded {len(models)} models")


def preprocess(df: pd.DataFrame) -> np.ndarray:
    data = df.copy()
    for col in ["Id", "SalePrice"]:
        if col in data.columns:
            data = data.drop(col, axis=1)

    for col in params["none_cols"]:
        if col in data.columns:
            data[col] = data[col].fillna("None")

    for col in params["zero_cols"]:
        if col in data.columns:
            data[col] = data[col].fillna(0)

    if "LotFrontage" in data.columns:
        if "Neighborhood" in data.columns:
            data["LotFrontage"] = (
                data.groupby("Neighborhood")["LotFrontage"]
                    .transform(lambda x: x.fillna(x.median()))
            )
        data["LotFrontage"] = data["LotFrontage"].fillna(data["LotFrontage"].median())

    for col in data.select_dtypes(include="object").columns:
        mode_val = data[col].mode()
        data[col] = data[col].fillna(mode_val[0] if not mode_val.empty else "None")
    data = data.fillna(0)

    data["TotalSF"]      = data.get("TotalBsmtSF", 0) + data.get("1stFlrSF", data.get("GrLivArea", 1500)) + data.get("2ndFlrSF", 0)
    data["HouseAge"]     = data.get("YrSold", 2010) - data.get("YearBuilt", 2000)
    data["RemodelAge"]   = data.get("YrSold", 2010) - data.get("YearRemodAdd", 2000)
    data["TotalBath"]    = (data.get("FullBath", 2) + 0.5 * data.get("HalfBath", 0)
                            + data.get("BsmtFullBath", 0) + 0.5 * data.get("BsmtHalfBath", 0))
    data["TotalPorchSF"] = (data.get("OpenPorchSF", 0) + data.get("EnclosedPorch", 0)
                            + data.get("3SsnPorch", 0) + data.get("ScreenPorch", 0))

    for col in data.select_dtypes(include="object").columns:
        if col in category_maps:
            data[col] = data[col].astype(str).map(category_maps[col]).fillna(0).astype(int)
        else:
            data[col] = 0

    for col in feature_names:
        if col not in data.columns:
            data[col] = 0
    return data[feature_names].values


def ensemble_predict(X: np.ndarray) -> np.ndarray:
    preds = np.column_stack([m.predict(X) for m in models.values()])
    return np.expm1(preds.mean(axis=1))


class HouseFeatures(BaseModel):
    """House features for price prediction. All fields are optional with sensible defaults."""
    MSSubClass:    Optional[int]   = Field(60,    description="Dwelling type code")
    MSZoning:      Optional[str]   = Field("RL",  description="Zoning classification")
    LotArea:       Optional[int]   = Field(9600,  description="Lot size in sq ft")
    LotFrontage:   Optional[float] = Field(70.0,  description="Linear feet of street")
    Neighborhood:  Optional[str]   = Field("CollgCr", description="Neighborhood name")
    OverallQual:   Optional[int]   = Field(6,     description="Overall quality 1-10")
    OverallCond:   Optional[int]   = Field(5,     description="Overall condition 1-10")
    YearBuilt:     Optional[int]   = Field(2000,  description="Construction year")
    YearRemodAdd:  Optional[int]   = Field(2000,  description="Remodel year")
    GrLivArea:     Optional[int]   = Field(1500,  description="Above-grade living area sq ft")
    TotalBsmtSF:   Optional[int]   = Field(800,   description="Total basement sq ft")
    GarageCars:    Optional[int]   = Field(2,     description="Garage capacity in cars")
    GarageArea:    Optional[int]   = Field(480,   description="Garage area sq ft")
    FullBath:      Optional[int]   = Field(2,     description="Full bathrooms")
    HalfBath:      Optional[int]   = Field(0,     description="Half bathrooms")
    BedroomAbvGr:  Optional[int]   = Field(3,     description="Bedrooms above grade")
    KitchenQual:   Optional[str]   = Field("TA",  description="Kitchen quality Ex/Gd/TA/Fa/Po")
    TotRmsAbvGrd:  Optional[int]   = Field(7,     description="Total rooms above grade")
    Fireplaces:    Optional[int]   = Field(1,     description="Number of fireplaces")
    MoSold:        Optional[int]   = Field(6,     description="Month sold 1-12")
    YrSold:        Optional[int]   = Field(2010,  description="Year sold")
    SaleType:      Optional[str]   = Field("WD",  description="Sale type")
    SaleCondition: Optional[str]   = Field("Normal", description="Sale condition")
    BsmtFullBath:  Optional[int]   = Field(0)
    BsmtHalfBath:  Optional[int]   = Field(0)
    OpenPorchSF:   Optional[int]   = Field(0)
    EnclosedPorch: Optional[int]   = Field(0)
    ScreenPorch:   Optional[int]   = Field(0)


class PredictionResponse(BaseModel):
    predicted_price: float
    formatted:       str
    model_used:      str
    confidence_note: str


@app.get("/", tags=["health"])
def root():
    return {
        "status":  "ok",
        "message": "House Price Prediction API is running",
        "docs":    "/docs",
        "models":  list(models.keys()),
    }


@app.get("/models", tags=["info"])
def list_models():
    cv_path = os.path.join(MODELS_DIR, "cv_results.json")
    with open(cv_path) as f:
        cv = json.load(f)
    return {"models": cv, "best": min(cv, key=lambda k: cv[k]["mean_rmse"])}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_single(house: HouseFeatures):
    """Predict sale price from JSON house features."""
    try:
        data = house.model_dump()
        data["1stFlrSF"] = data.get("GrLivArea", 1500)
        data["2ndFlrSF"] = 0
        data["3SsnPorch"] = 0
        df = pd.DataFrame([data])
        X = preprocess(df)
        price = float(ensemble_predict(X)[0])
        return PredictionResponse(
            predicted_price=round(price, 2),
            formatted=f"${price:,.0f}",
            model_used="ensemble (ridge + lasso + rf + gbm)",
            confidence_note="+-15% typical range based on CV RMSE ~0.13 on log scale",
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/csv", tags=["prediction"])
async def predict_csv(file: UploadFile = File(...)):
    """Upload a CSV (same format as test.csv) and get back predictions."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    ids = df["Id"].tolist() if "Id" in df.columns else list(range(len(df)))
    X = preprocess(df)
    prices = ensemble_predict(X)
    return {
        "count": len(prices),
        "predictions": [
            {"Id": int(i), "predicted_price": round(float(p), 2), "formatted": f"${p:,.0f}"}
            for i, p in zip(ids, prices)
        ],
    }
