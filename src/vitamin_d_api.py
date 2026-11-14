
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict


router = APIRouter(tags=["Vitamin D Research ML"])

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "vitamin_d_final_research_model.joblib"
)


class VitaminDPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age_years: float = Field(ge=18, le=120)
    bmi: Optional[float] = Field(default=None, gt=0, le=100)

    avg_energy_kcal: float = Field(ge=0)
    avg_protein_g: float = Field(ge=0)
    avg_fat_g: float = Field(ge=0)
    avg_calcium_mg: float = Field(ge=0)
    avg_vitamin_d_mcg: float = Field(ge=0)
    avg_magnesium_mg: float = Field(ge=0)
    avg_zinc_mg: float = Field(ge=0)

    sex_code: int = Field(ge=1, le=2)
    examination_period: int = Field(ge=1, le=2)

    supplement_count_30day: Optional[float] = Field(
        default=None, ge=0
    )
    vitamin_d_supplement_30day_mcg: Optional[float] = Field(
        default=None, ge=0
    )


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Research model not found: {MODEL_PATH}"
        )

    # Only load model artifacts created by your own
    # trusted training script.
    return joblib.load(MODEL_PATH)


@router.post("/predict-vitamin-d")
def predict_vitamin_d(request: VitaminDPredictionRequest):
    try:
        artifact = load_model()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Vitamin D research model is unavailable."
        ) from exc

    data = request.model_dump()

    count = data["supplement_count_30day"]
    amount = data["vitamin_d_supplement_30day_mcg"]

    if count is not None and count == 0:
        if amount is not None and amount > 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Vitamin D supplement intake cannot be "
                    "positive when supplement count is zero."
                ),
            )

    # Match training-time feature engineering.
    data["vitamin_d_30day_missing"] = int(amount is None)

    if count == 0 and amount is None:
        data["vitamin_d_supplement_30day_mcg"] = 0.0

    for key, value in data.items():
        if isinstance(value, (float, int)) and not np.isfinite(value):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid numeric value: {key}"
            )

    frame = pd.DataFrame(
        [data],
        columns=artifact["features"]
    )

    probability = float(
        artifact["model"].predict_proba(frame)[0, 1]
    )

    threshold = float(artifact["threshold"])

    return {
        "research_outcome": (
            "Serum 25-hydroxyvitamin D below 50 nmol/L"
        ),
        "estimated_probability": round(probability, 4),
        "screen_positive": probability >= threshold,
        "classification_threshold": threshold,
        "model": "Logistic Regression",
        "population": "NHANES 2017-2018 US adults",
        "disclaimer": (
            "Experimental statistical estimate only. "
            "Not a diagnosis, not clinically validated, "
            "and not validated for astronauts. "
            "Do not use for medical decisions."
        ),
    }
