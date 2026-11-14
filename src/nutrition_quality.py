"""Read-only V3 audit and feasibility reports."""
import json
from pathlib import Path
import pandas as pd
from src.nutrition_engine_v2 import DOMAIN_FEATURES

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "processed" / "astronaut_nutrition_features_v3.csv"
REPORTS = ROOT / "reports"

SOURCE_MAPPINGS = {
    "serum_glucose_mg_dl": "BIOPRO__LBXSGL", "hba1c_percent": "GHB__LBXGH",
    "ferritin_ng_ml": "FERTIN__LBXFER", "serum_iron_ug_dl": "BIOPRO__LBXSIR",
    "serum_creatinine_mg_dl": "BIOPRO__LBXSCR", "bun_mg_dl": "BIOPRO__LBXSBU",
    "urine_albumin_ug_ml": "ALB_CR__URXUMA", "urine_creatinine_mg_dl": "ALB_CR__URXUCR",
    "alt_u_l": "BIOPRO__LBXSATSI", "ast_u_l": "BIOPRO__LBXSASSI", "albumin_g_dl": "BIOPRO__LBXSAL",
    "total_bilirubin_mg_dl": "BIOPRO__LBXSTB", "hdl_cholesterol_mg_dl": "HDL__LBDHDD",
    "total_cholesterol_mg_dl": "TCHOL__LBXTC", "triglycerides_mg_dl": "TRIGLY__LBXTR",
    "sodium_mmol_l": "BIOPRO__LBXSNASI", "potassium_mmol_l": "BIOPRO__LBXSKSI",
    "chloride_mmol_l": "BIOPRO__LBXSCLSI", "serum_calcium_mg_dl": "BIOPRO__LBXSCA",
    "serum_phosphorus_mg_dl": "BIOPRO__LBXSPH",
}


def build_quality_report(path=DATASET):
    df = pd.read_csv(path, low_memory=False)
    numeric = df.select_dtypes(include="number")
    domain = {}
    for name, fields in DOMAIN_FEATURES.items():
        available = [field for field in fields if field in df]
        complete = df[available].notna().all(axis=1) if available else pd.Series(False, index=df.index)
        domain[name] = {"features": fields, "available_features": available,
                        "complete_participants": int(complete.sum()), "complete_percent": round(float(complete.mean() * 100), 2),
                        "joint_availability": int(df[available].notna().all(axis=1).sum()) if available else 0}
    report = {"dataset": Path(path).name, "participant_count": int(len(df)), "column_count": int(len(df.columns)),
              "unique_participant_ids": int(df["SEQN"].nunique()) if "SEQN" in df else None,
              "duplicate_records": int(df.duplicated().sum()), "duplicate_columns": int(df.columns.duplicated().sum()),
              "dtypes": {str(k): int(v) for k, v in df.dtypes.astype(str).value_counts().items()},
              "feature_missingness": {key: {"missing": int(value), "percent": round(value / len(df) * 100, 2)} for key, value in df.isna().sum().items()},
              "numeric_ranges": {key: {"min": None if pd.isna(value.min()) else float(value.min()), "max": None if pd.isna(value.max()) else float(value.max())} for key, value in numeric.items()},
              "domain_completeness": domain,
              "mapping_verification": {name: {"source_column": source, "present": source in df.columns, "status": "copied_without_unit_conversion; official NHANES documentation still required"} for name, source in SOURCE_MAPPINGS.items()},
              "blood_pressure": {"method": "mean of valid positive readings; systolic <=300 and diastolic <=200 are engineering filters", "source_fields": ["BPX__BPXSY1..4", "BPX__BPXDI1..4"]},
              "limitations": ["Read-only audit; ranges are descriptive, not clinical plausibility decisions.", "Units require official NHANES documentation verification before clinical use."]}
    return report


def build_ml_feasibility(report):
    domains = {}
    for name, coverage in report["domain_completeness"].items():
        domains[name] = {"candidate_targets": [], "observed_or_derived": "No domain-specific outcome target established",
                         "usable_complete_samples": coverage["complete_participants"], "missingness_constraint": coverage["complete_percent"],
                         "leakage_risk": "High if a target is constructed from included biomarker features", "class_balance": "Not applicable without a legitimate target",
                         "meaningful_supervised_validation": False,
                         "limitation": "No verified clinical or longitudinal outcome target; do not train a domain predictor."}
    return {"dataset": report["dataset"], "domains": domains,
            "existing_research_artifacts": "Existing vitamin-D artifacts are separate research work and not evidence for nine-domain or astronaut prediction.",
            "pipeline_guidance": ["Use train/test splits before preprocessing.", "Fit imputers/scalers on training data only through sklearn Pipeline.", "Record seed, selected features, target provenance, and leakage audit."],
            "conclusion": "No new supervised nutrition-domain model trained because legitimate targets were not established."}


def write_reports():
    REPORTS.mkdir(exist_ok=True)
    quality = build_quality_report()
    feasibility = build_ml_feasibility(quality)
    for filename, data in (("nutrition_v3_quality_report.json", quality), ("nutrition_ml_feasibility.json", feasibility)):
        with (REPORTS / filename).open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, allow_nan=False)
    return quality, feasibility
