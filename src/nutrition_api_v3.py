from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from src.nutrition_engine_v2 import assess_nutrition
from src.nutrition_dietary import assess_dietary_intake
from src.nutrition_longitudinal import summarize_observations
from src.nutrition_insights import generate_insights
from src.nutrition_quality import build_quality_report, build_ml_feasibility
from src.nutrition_reporting import serialize_research_report

router = APIRouter(prefix="/nutrition-v3", tags=["Astronaut Nutrition ML V3 Research"])

class DietaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intakes: Dict[str, float] = Field(min_length=1)
    references: Dict[str, float] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)

class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject_id: str = Field(min_length=1)
    timestamp: datetime
    measurements: Dict[str, float] = Field(default_factory=dict)
    units: Dict[str, str] = Field(default_factory=dict)
    source_metadata: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)

class LongitudinalRequest(BaseModel):
    subject_id: str = Field(min_length=1)
    observations: List[Observation] = Field(min_length=1)

class InsightRequest(BaseModel):
    measurements: Dict[str, float] = Field(min_length=1)
    longitudinal: Optional[LongitudinalRequest] = None

@router.get("/data-quality")
def data_quality():
    return build_quality_report()

@router.get("/ml-feasibility")
def ml_feasibility():
    return build_ml_feasibility(build_quality_report())

@router.post("/dietary-assess")
def dietary_assess(request: DietaryRequest):
    try:
        return assess_dietary_intake(request.intakes, request.references, request.context)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@router.post("/longitudinal-assess")
def longitudinal_assess(request: LongitudinalRequest):
    try:
        return summarize_observations(request.subject_id, [item.model_dump(mode="json") for item in request.observations])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@router.post("/insights")
def insights(request: InsightRequest):
    assessment = assess_nutrition(request.measurements)
    longitudinal = None
    if request.longitudinal:
        longitudinal = summarize_observations(request.longitudinal.subject_id, [item.model_dump(mode="json") for item in request.longitudinal.observations])
    return generate_insights(assessment, longitudinal)

@router.post("/research-report")
def research_report(request: InsightRequest):
    assessment = assess_nutrition(request.measurements)
    longitudinal = None
    if request.longitudinal:
        longitudinal = summarize_observations(request.longitudinal.subject_id, [item.model_dump(mode="json") for item in request.longitudinal.observations])
    return serialize_research_report(assessment, longitudinal=longitudinal, feasibility=build_ml_feasibility(build_quality_report()))
