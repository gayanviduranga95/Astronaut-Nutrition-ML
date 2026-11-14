import math

import pytest
from fastapi.testclient import TestClient

from api import app
from src.nutrition_engine_v2 import DOMAIN_FEATURES, assess_nutrition
from src.nutrition_interpretation import (
    calculate_urine_acr,
    interpret_nutrition,
)


def complete_record():
    return {
        "avg_protein_g": 80, "avg_iron_mg": 14, "bmi": 23.5,
        "albumin_g_dl": 4.2, "hemoglobin_g_dl": 13.5, "mcv_fl": 90,
        "ferritin_ng_ml": 45, "serum_iron_ug_dl": 100,
        "serum_glucose_mg_dl": 95, "hba1c_percent": 5.7,
        "serum_creatinine_mg_dl": 0.9, "bun_mg_dl": 15,
        "urine_albumin_ug_ml": 15, "urine_creatinine_mg_dl": 100,
        "serum_calcium_mg_dl": 9.5, "serum_phosphorus_mg_dl": 3.5,
        "alt_u_l": 20, "ast_u_l": 22, "total_bilirubin_mg_dl": 0.7,
        "total_cholesterol_mg_dl": 200, "hdl_cholesterol_mg_dl": 50,
        "triglycerides_mg_dl": 150, "systolic_bp_mean_mmhg": 120,
        "diastolic_bp_mean_mmhg": 80, "sodium_mmol_l": 140,
        "potassium_mmol_l": 4.2, "chloride_mmol_l": 103,
        "serum_vitamin_d_nmol_l": 60,
    }


def test_all_nine_domains_have_descriptive_summaries_and_limitations():
    result = assess_nutrition(complete_record())

    assert set(result["domains"]) == set(DOMAIN_FEATURES)
    for domain, assessment in result["domains"].items():
        assert assessment["measurement_summaries"], domain
        assert assessment["interpretation_summary"]
        assert assessment["limitation"]

    assert result["domains"]["kidney_stone"]["status"] == "limited_assessment"
    assert result["domains"]["hydration_electrolytes"]["status"] == "limited_assessment"
    assert result["domains"]["bone_nutrition"]["status"] == "limited_assessment"


def test_missing_measurements_preserve_existing_status_and_explain_limit():
    result = assess_nutrition({"hba1c_percent": 5.6})

    assert result["domains"]["glucose_regulation"]["status"] == "insufficient_data"
    assert "serum_glucose_mg_dl" in result["domains"]["glucose_regulation"]["missing_features"]
    assert "Incomplete measurements limit interpretation" in result["domains"]["glucose_regulation"]["interpretation_summary"]
    assert result["domains"]["anemia_iron"]["status"] == "insufficient_data"
    assert "No supported measurements" in result["domains"]["anemia_iron"]["interpretation_summary"]


@pytest.mark.parametrize(
    ("measurement", "value", "expected"),
    [
        ("hba1c_percent", 5.699, "below_prediabetes_threshold"),
        ("hba1c_percent", 5.7, "prediabetes_range"),
        ("hba1c_percent", 6.5, "diabetes_range"),
        ("total_cholesterol_mg_dl", 199.9, "desirable_range"),
        ("total_cholesterol_mg_dl", 200, "borderline_high_range"),
        ("total_cholesterol_mg_dl", 240, "high_range"),
        ("triglycerides_mg_dl", 149.9, "normal_range"),
        ("triglycerides_mg_dl", 150, "borderline_high_range"),
        ("triglycerides_mg_dl", 200, "high_or_above_range"),
    ],
)
def test_reference_comparison_boundaries(measurement, value, expected):
    result = interpret_nutrition({measurement: value})
    assert result["findings"][0]["interpretation"] == expected


@pytest.mark.parametrize("value", [None, True, "invalid", math.nan, math.inf])
def test_invalid_and_non_finite_measurements_are_not_interpreted(value):
    result = interpret_nutrition({"hba1c_percent": value})
    assert result["findings"] == []
    assert result["measurement_summaries"] == []


def test_urine_acr_uses_verified_unit_conversion_and_rejects_bad_denominators():
    # ug/mL albumin equals mg/L; mg/dL creatinine equals 0.01 g/L,
    # so 15 / 1 = 15 mg/g for albumin=15 ug/mL and creatinine=100 mg/dL.
    measurements = {"urine_albumin_ug_ml": 15, "urine_creatinine_mg_dl": 100}
    assert calculate_urine_acr(measurements) == 15
    assert calculate_urine_acr({**measurements, "urine_creatinine_mg_dl": 0}) is None
    assert calculate_urine_acr({**measurements, "urine_creatinine_mg_dl": -1}) is None
    assert calculate_urine_acr({**measurements, "urine_creatinine_mg_dl": math.nan}) is None

    result = assess_nutrition(measurements)
    assert result["interpretation"]["derived_measurements"] == {"urine_acr_mg_g": 15.0}
    assert result["domains"]["kidney_health"]["derived_measurements"] == {"urine_acr_mg_g": 15.0}


def test_findings_are_grouped_once_under_the_correct_domain():
    result = assess_nutrition({
        "hba1c_percent": 6.5,
        "total_cholesterol_mg_dl": 240,
        "triglycerides_mg_dl": 200,
    })

    assert [item["measurement"] for item in result["domains"]["glucose_regulation"]["findings"]] == ["hba1c_percent"]
    assert [item["measurement"] for item in result["domains"]["cardiovascular_metabolic"]["findings"]] == [
        "total_cholesterol_mg_dl", "triglycerides_mg_dl",
    ]
    assert sum(domain["finding_count"] for domain in result["domains"].values()) == 3


def test_backwards_compatible_response_fields_are_retained():
    result = assess_nutrition({"hba1c_percent": 5.9})
    assert {"system", "engine_version", "assessment_type", "domains", "interpretation", "disclaimer"} <= set(result)
    assert {"status", "available_count", "required_count", "measurements", "missing_features", "findings", "finding_count"} <= set(result["domains"]["glucose_regulation"])
    assert {"interpretation_version", "findings", "finding_count", "disclaimer"} <= set(result["interpretation"])


def test_api_health_success_validation_and_existing_routes():
    client = TestClient(app)

    assert client.get("/nutrition-v2/health").json()["domains"] == 9
    successful = client.post("/nutrition-v2/assess", json={"measurements": {"hba1c_percent": 5.9}})
    assert successful.status_code == 200
    assert successful.json()["domains"]["glucose_regulation"]["finding_count"] == 1

    invalid = client.post("/nutrition-v2/assess", json={"measurements": {"unknown": 1}})
    assert invalid.status_code == 422
    non_finite = client.post("/nutrition-v2/assess", json={"measurements": {"hba1c_percent": "NaN"}})
    assert non_finite.status_code == 422

    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200
    legacy_assessment = client.post("/assess-nutrition", json={
        "intakes": {"protein_g": 60},
        "requirements": {"protein_g": 75},
    })
    assert legacy_assessment.status_code == 200
    assert legacy_assessment.json()["assessment_type"] == "nutrient_adequacy_screening"
