
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

INPUT = DATA / "astronaut_nutrition_master_v2.csv"
OUTPUT = REPORTS / "lab_feature_audit.json"

REPORTS.mkdir(parents=True, exist_ok=True)

# Candidate source columns.
# These must be confirmed against official NHANES
# documentation before clinical interpretation.
CANDIDATES = {
    "glucose_regulation": {
        "glucose": ("BIOPRO", "LBXSGL"),
        "hba1c": ("GHB", "LBXGH"),
    },
    "anemia_iron": {
        "hemoglobin": ("CBC", "LBXHGB"),
        "mcv": ("CBC", "LBXMCVSI"),
        "ferritin": ("FERTIN", "LBXFER"),
        "serum_iron": ("BIOPRO", "LBXSIR"),
    },
    "kidney_health": {
        "creatinine": ("BIOPRO", "LBXSCR"),
        "bun": ("BIOPRO", "LBXSBU"),
        "urine_albumin": ("ALB_CR", "URXUMA"),
        "urine_creatinine": ("ALB_CR", "URXUCR"),
    },
    "liver_health": {
        "alt": ("BIOPRO", "LBXSATSI"),
        "ast": ("BIOPRO", "LBXSASSI"),
        "albumin": ("BIOPRO", "LBXSAL"),
        "bilirubin": ("BIOPRO", "LBXSTB"),
    },
    "cardiovascular_metabolic": {
        "hdl": ("HDL", "LBDHDD"),
        "total_cholesterol": ("TCHOL", "LBXTC"),
        "triglycerides": ("TRIGLY", "LBXTR"),
    },
    "hydration_electrolytes": {
        "sodium": ("BIOPRO", "LBXSNASI"),
        "potassium": ("BIOPRO", "LBXSKSI"),
        "chloride": ("BIOPRO", "LBXSCLSI"),
    },
    "bone_nutrition": {
        "vitamin_d": ("VID", "LBXVIDMS"),
        "calcium": ("BIOPRO", "LBXSCA"),
        "phosphorus": ("BIOPRO", "LBXSPH"),
    },
}

SOURCES = [
    "BIOPRO", "GHB", "FERTIN", "HDL",
    "TCHOL", "TRIGLY", "ALB_CR", "BPX"
]

print("=" * 65)
print("STEP 38: LABORATORY FEATURE AUDIT")
print("=" * 65)

df = pd.read_csv(INPUT, low_memory=False)

print("Participants:", len(df))
print("Columns:", len(df.columns))

report = {
    "rows": int(len(df)),
    "columns": int(len(df.columns)),
    "candidate_features": {},
    "source_columns": {},
}

print("\nCANDIDATE FEATURE AVAILABILITY")

for module, features in CANDIDATES.items():
    print(f"\n[{module}]")
    report["candidate_features"][module] = {}

    for feature, (source, code) in features.items():
        column = f"{source}__{code}"

        if column in df.columns:
            values = pd.to_numeric(
                df[column], errors="coerce"
            )

            count = int(values.notna().sum())
            status = "FOUND"

            print(
                f"  {feature}: {column} "
                f"({count} available)"
            )

        else:
            count = 0
            status = "NOT_FOUND"

            print(
                f"  {feature}: {column} NOT FOUND"
            )

        report["candidate_features"][module][feature] = {
            "column": column,
            "status": status,
            "non_missing": count,
        }

print("\n" + "=" * 65)
print("ALL NEW SOURCE COLUMNS")
print("=" * 65)

for source in SOURCES:
    prefix = f"{source}__"

    columns = [
        col for col in df.columns
        if col.startswith(prefix)
    ]

    print(f"\n{source}: {len(columns)} columns")

    report["source_columns"][source] = {}

    for column in columns:
        count = int(df[column].notna().sum())

        report["source_columns"][source][column] = count

        print(f"  {column}: {count}")

with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 65)
print("AUDIT COMPLETE")
print("=" * 65)
print("Report saved:", OUTPUT)

