from __future__ import annotations

import pathlib
from typing import Tuple, List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "datasets"
BANK_CANDIDATES: List[pathlib.Path] = [
    DATA_DIR / "bank.csv",
    DATA_DIR / "bank-full.csv",
    DATA_DIR / "bank_marketing.csv",
    DATA_DIR / "bank-additional-full.csv",
]


def load_bank_duration_pdays() -> Tuple[pd.DataFrame, pd.Series]:
    """Load Bank Marketing CSV and return X(df with duration,pdays) and y(series 0/1)."""
    for csv_path in BANK_CANDIDATES:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            break
    else:
        raise FileNotFoundError(
            f"Bank Marketing CSV not found in {DATA_DIR}. Expected one of: "
            + ", ".join(p.name for p in BANK_CANDIDATES)
        )

    df = df.rename(columns={c: c.lower() for c in df.columns})

    # Target detection and mapping to {0,1}
    if "y" in df.columns:
        target_col = "y"
    elif "deposit" in df.columns:
        target_col = "deposit"
    elif "target" in df.columns:
        target_col = "target"
    else:
        target_col = df.columns[-1]

    y_raw = df[target_col]
    if y_raw.dtype == object:
        y = y_raw.astype(str).str.lower().map({"yes": 1, "no": 0})
        if y.isna().any():
            classes = sorted(y_raw.dropna().astype(str).str.lower().unique())
            mapping = {cls: i for i, cls in enumerate(classes[:2])}
            y = y_raw.astype(str).str.lower().map(mapping)
    else:
        classes = np.sort(y_raw.unique())
        mapping = {cls: i for i, cls in enumerate(classes[:2])}
        y = y_raw.map(mapping)
    y = y.astype(int)

    # Fixed numeric features
    feature_names = ("duration", "pdays")
    for col in feature_names:
        if col not in df.columns:
            raise ValueError(f"Required feature '{col}' not present in CSV")

    X = df[list(feature_names)].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def build_pipeline() -> Pipeline:
    """Create the scaler + classifier pipeline.

    We choose LogisticRegression for a fast, stable classifier. Class weights are
    balanced to mitigate target imbalance common in this dataset.
    """
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
    return pipe


class PredictRequest(BaseModel):
    duration: float = Field(..., ge=0, description="Last contact duration in seconds")
    pdays: float = Field(..., ge=-1, description="Days since last contact (-1 means never)")


class PredictResponse(BaseModel):
    prediction: str
    probability_yes: float
    features: dict


app = FastAPI(title="Bank Marketing Classifier", version="1.0.0")


# Global model object
MODEL: Pipeline | None = None
FEATURE_ORDER = ("duration", "pdays")


@app.on_event("startup")
def _startup() -> None:
    global MODEL
    X, y = load_bank_duration_pdays()
    model = build_pipeline()
    model.fit(X.to_numpy(dtype=float), y.to_numpy())
    MODEL = model


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    assert MODEL is not None, "Model is not initialized"
    features = np.array([[req.duration, req.pdays]], dtype=float)
    proba_yes = float(MODEL.predict_proba(features)[:, 1][0])
    pred = "yes" if proba_yes >= 0.5 else "no"
    return PredictResponse(
        prediction=pred,
        probability_yes=proba_yes,
        features={"duration": req.duration, "pdays": req.pdays},
    )


@app.get("/")
def root() -> dict:
    return {
        "message": "Bank Marketing API. Use /docs for Swagger UI.",
        "features": list(FEATURE_ORDER),
        "predict_example": {"duration": 120.0, "pdays": 10},
    }


