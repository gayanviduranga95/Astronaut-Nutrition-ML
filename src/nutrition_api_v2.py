
import math
from typing import Dict

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

try:
    from src.nutrition_engine_v2 import (
        assess_nutrition,
        DOMAIN_FEATURES,
    )
except ModuleNotFoundError:
    from nutrition_engine_v2 import (
        assess_nutrition,
        DOMAIN_FEATURES,
    )


router = APIRouter(
    prefix="/nutrition-v2",
    tags=["Astronaut Nutrition ML V2"],
)

ALLOWED_FEATURES = {
    feature
    for features in DOMAIN_FEATURES.values()
    for feature in features
}

# Broad input sanity limits only.
# NOT diagnostic or astronaut-specific clinical thresholds.
FEATURE_LIMITS = {
    "bmi": (1, 100),
    "hba1c_percent": (0, 25),
    "serum_glucose_mg_dl": (0, 1000),
    "sodium_mmol_l": (0, 250),
    "potassium_mmol_l": (0, 20),
    "chloride_mmol_l": (0, 200),
    "systolic_bp_mean_mmhg": (0, 300),
    "diastolic_bp_mean_mmhg": (0, 200),
}

DEFAULT_LIMITS = (0, 100000)


class NutritionAssessmentRequest(BaseModel):
    measurements: Dict[str, float] = Field(
        ...,
        min_length=1,
        description="Recognized nutrition and laboratory measurements",
    )

    @field_validator("measurements", mode="before")
    @classmethod
    def validate_measurements(cls, values):
        if not isinstance(values, dict):
            raise ValueError(
                "measurements must be a JSON object"
            )

        if not values:
            raise ValueError(
                "At least one measurement is required"
            )

        cleaned = {}

        for name, value in values.items():
            if name not in ALLOWED_FEATURES:
                raise ValueError(
                    f"Unknown measurement: {name}"
                )

            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise ValueError(
                    f"{name} must be a JSON number"
                )

            number = float(value)

            if not math.isfinite(number):
                raise ValueError(
                    f"{name} must be finite"
                )

            lower, upper = FEATURE_LIMITS.get(
                name, DEFAULT_LIMITS
            )

            if not lower <= number <= upper:
                raise ValueError(
                    f"{name} is outside the accepted "
                    f"input range [{lower}, {upper}]"
                )

            cleaned[name] = number

        return cleaned


@router.get("/health")
def nutrition_health():
    return {
        "status": "ok",
        "engine": "Astronaut Nutrition ML V2",
        "domains": len(DOMAIN_FEATURES),
        "accepted_measurements": len(ALLOWED_FEATURES),
    }


@router.post("/assess")
def nutrition_assess(
    request: NutritionAssessmentRequest,
):
    return assess_nutrition(request.measurements)
