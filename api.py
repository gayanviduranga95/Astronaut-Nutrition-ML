
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.nutrition_assessment import assess_nutrition
from src.vitamin_d_api import router as vitamin_d_router
from src.multi_nutrient_api import router as multi_nutrient_router
from src.nutrition_api_v2 import router as nutrition_v2_router
app = FastAPI(
    title="Astronaut Nutrition Assessment API",
    version="1.1.0"
)

# Development only. Restrict origins for deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

class NutritionRequest(BaseModel):
    intakes: Dict[str, float] = Field(min_length=1)
    requirements: Dict[str, float] = Field(min_length=1)


@app.get("/")
def home():
    return {
        "service": "Astronaut Nutrition Assessment",
        "status": "running",
        "version": "1.1.0"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/assess-nutrition")
def assess(request: NutritionRequest):
    try:
        return assess_nutrition(
            intakes=request.intakes,
            requirements=request.requirements
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        )
app.include_router(vitamin_d_router)
app.include_router(multi_nutrient_router)
app.include_router(nutrition_v2_router)