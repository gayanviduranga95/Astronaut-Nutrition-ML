import json
from pathlib import Path

def serialize_research_report(assessment, dietary=None, longitudinal=None, feasibility=None):
    report = {"report_type": "astronaut_nutrition_research_report", "laboratory_observations": assessment,
              "dietary_observations": dietary, "longitudinal_changes": longitudinal,
              "ml_feasibility": feasibility,
              "limitations": ["NHANES is an Earth-population cross-sectional dataset, not astronaut observations.", "No diagnostic, clinical-ready, or astronaut-validated predictions are included."]}
    json.dumps(report, allow_nan=False)
    return report

def write_json_report(report, path):
    path = Path(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
