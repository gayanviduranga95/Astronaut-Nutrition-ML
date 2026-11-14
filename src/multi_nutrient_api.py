
from typing import Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from src.multi_nutrient_engine import assess_multi_nutrient

router = APIRouter(tags=["Astronaut Nutrition"])


class MultiNutrientRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intakes: Dict[str, float]
    targets: Dict[str, float]


@router.post("/assess-multi-nutrient")
def assess_nutrition(request: MultiNutrientRequest):
    try:
        return assess_multi_nutrient(
            request.intakes,
            request.targets
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc
