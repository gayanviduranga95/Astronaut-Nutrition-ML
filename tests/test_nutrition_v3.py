import math
from fastapi.testclient import TestClient
from api import app
from src.nutrition_dietary import assess_dietary_intake
from src.nutrition_longitudinal import summarize_observations
from src.nutrition_quality import build_ml_feasibility
from src.nutrition_reporting import serialize_research_report
from src.nutrition_validation import validate_measurements, valid_denominator
from src.nutrition_ml_pipeline import build_research_classifier, model_metadata


def test_validation_rejects_nonfinite_and_engineering_outliers():
    cleaned, issues = validate_measurements({"hba1c_percent": math.nan, "bmi": 101, "sodium_mmol_l": 140})
    assert cleaned == {"sodium_mmol_l": 140.0}
    assert len(issues) == 2
    assert not valid_denominator(0)


def test_dietary_comparison_requires_context_and_preserves_observation():
    no_context = assess_dietary_intake({"protein_g": 50}, {"protein_g": 75})
    assert no_context["results"][0]["percentage_of_reference"] is None
    result = assess_dietary_intake({"protein_g": 50, "iron_mg": 10}, {"protein_g": 75}, {"age_years": 30})
    assert result["results"][0]["percentage_of_reference"] == 66.67
    assert result["results"][1]["comparison_status"] == "insufficient_reference_context"


def test_longitudinal_changes_duplicate_timestamps_and_zero_baseline():
    rows = [
        {"subject_id": "subject-a", "timestamp": "2026-01-01T00:00:00Z", "measurements": {"hba1c_percent": 0, "sodium_mmol_l": 140}},
        {"subject_id": "subject-a", "timestamp": "2026-01-11T00:00:00Z", "measurements": {"hba1c_percent": 2, "sodium_mmol_l": 142}},
    ]
    summary = summarize_observations("subject-a", rows)
    hba1c = next(item for item in summary["trends"] if item["measurement"] == "hba1c_percent")
    assert hba1c["percentage_change"] is None
    assert hba1c["interval_days"] == 10
    try:
        summarize_observations("subject-a", [rows[0], rows[0]])
    except ValueError as exc:
        assert "Duplicate" in str(exc)
    else:
        raise AssertionError("duplicate timestamps must be rejected")


def test_feasibility_and_serializable_report_do_not_create_predictions():
    feasibility = build_ml_feasibility({"dataset": "synthetic.csv", "domain_completeness": {"glucose_regulation": {"complete_participants": 2, "complete_percent": 50}}})
    assert not feasibility["domains"]["glucose_regulation"]["meaningful_supervised_validation"]
    report = serialize_research_report({"domains": {}}, feasibility=feasibility)
    assert "NHANES" in report["limitations"][0]


def test_pipeline_requires_target_provenance_and_excludes_target_feature():
    try:
        build_research_classifier(["outcome"], [], {"target_name": "outcome"})
    except ValueError as exc:
        assert "Target" in str(exc)
    else:
        raise AssertionError("target leakage must be rejected")
    pipeline = build_research_classifier(["feature"], [], {"target_name": "outcome", "source": "verified study outcome"})
    assert "preprocessor" in pipeline.named_steps
    assert not model_metadata(["feature"], {"target_name": "outcome"})["external_or_astronaut_validation"]


def test_v3_api_endpoints_and_existing_compatibility():
    client = TestClient(app)
    quality = client.get("/nutrition-v3/data-quality")
    assert quality.status_code == 200
    assert quality.json()["participant_count"] == 9254
    assert client.get("/nutrition-v3/ml-feasibility").status_code == 200
    dietary = client.post("/nutrition-v3/dietary-assess", json={"intakes": {"protein_g": 50}, "references": {"protein_g": 75}, "context": {"age_years": 30}})
    assert dietary.status_code == 200
    longitudinal = client.post("/nutrition-v3/longitudinal-assess", json={"subject_id": "a", "observations": [
        {"subject_id": "a", "timestamp": "2026-01-01T00:00:00Z", "measurements": {"hba1c_percent": 5.5}},
        {"subject_id": "a", "timestamp": "2026-02-01T00:00:00Z", "measurements": {"hba1c_percent": 5.7}},
    ]})
    assert longitudinal.status_code == 200
    assert client.post("/nutrition-v2/assess", json={"measurements": {"hba1c_percent": 5.9}}).status_code == 200
