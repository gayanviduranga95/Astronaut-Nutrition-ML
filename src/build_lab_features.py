
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

INPUT = PROCESSED / "astronaut_nutrition_master_v2.csv"
OUTPUT = PROCESSED / "astronaut_nutrition_features_v3.csv"
REPORT = REPORTS / "lab_features_v3_quality.json"

REPORTS.mkdir(parents=True, exist_ok=True)

# Map standardized feature names to NHANES source columns.
# Units should be checked against the official documentation
# before using the values for clinical interpretation.

FEATURE_MAP = {
    # Glucose regulation
    "serum_glucose_mg_dl": "BIOPRO__LBXSGL",
    "hba1c_percent": "GHB__LBXGH",

    # Anemia and iron
    "ferritin_ng_ml": "FERTIN__LBXFER",
    "serum_iron_ug_dl": "BIOPRO__LBXSIR",

    # Kidney
    "serum_creatinine_mg_dl": "BIOPRO__LBXSCR",
    "bun_mg_dl": "BIOPRO__LBXSBU",
    "urine_albumin_ug_ml": "ALB_CR__URXUMA",
    "urine_creatinine_mg_dl": "ALB_CR__URXUCR",

    # Liver
    "alt_u_l": "BIOPRO__LBXSATSI",
    "ast_u_l": "BIOPRO__LBXSASSI",
    "albumin_g_dl": "BIOPRO__LBXSAL",
    "total_bilirubin_mg_dl": "BIOPRO__LBXSTB",
    "alkaline_phosphatase_u_l": "BIOPRO__LBXSAPSI",
    "ggt_u_l": "BIOPRO__LBXSGTSI",

    # Cardiovascular metabolic
    "hdl_cholesterol_mg_dl": "HDL__LBDHDD",
    "total_cholesterol_mg_dl": "TCHOL__LBXTC",
    "triglycerides_mg_dl": "TRIGLY__LBXTR",

    # Electrolytes
    "sodium_mmol_l": "BIOPRO__LBXSNASI",
    "potassium_mmol_l": "BIOPRO__LBXSKSI",
    "chloride_mmol_l": "BIOPRO__LBXSCLSI",
    "bicarbonate_mmol_l": "BIOPRO__LBXSC3SI",

    # Bone-related nutrition
    "serum_calcium_mg_dl": "BIOPRO__LBXSCA",
    "serum_phosphorus_mg_dl": "BIOPRO__LBXSPH",

    # Protein nutrition
    "total_protein_g_dl": "BIOPRO__LBXSTP",
}

MODULE_FEATURES = {
    "nutritional_status": [
        "avg_protein_g",
        "avg_iron_mg",
        "bmi",
        "albumin_g_dl",
    ],
    "anemia_iron": [
        "hemoglobin_g_dl",
        "mcv_fl",
        "ferritin_ng_ml",
        "serum_iron_ug_dl",
    ],
    "glucose_regulation": [
        "serum_glucose_mg_dl",
        "hba1c_percent",
    ],
    "kidney_health": [
        "serum_creatinine_mg_dl",
        "bun_mg_dl",
        "urine_albumin_ug_ml",
        "urine_creatinine_mg_dl",
    ],
    "kidney_stone": [
        "serum_calcium_mg_dl",
        "serum_phosphorus_mg_dl",
    ],
    "liver_health": [
        "alt_u_l",
        "ast_u_l",
        "albumin_g_dl",
        "total_bilirubin_mg_dl",
    ],
    "cardiovascular_metabolic": [
        "total_cholesterol_mg_dl",
        "hdl_cholesterol_mg_dl",
        "triglycerides_mg_dl",
        "systolic_bp_mean_mmhg",
        "diastolic_bp_mean_mmhg",
    ],
    "hydration_electrolytes": [
        "sodium_mmol_l",
        "potassium_mmol_l",
        "chloride_mmol_l",
    ],
    "bone_nutrition": [
        "serum_vitamin_d_nmol_l",
        "serum_calcium_mg_dl",
        "serum_phosphorus_mg_dl",
    ],
}

print("=" * 65)
print("STEP 39: STANDARDIZED LAB FEATURE BUILD")
print("=" * 65)

df = pd.read_csv(INPUT, low_memory=False)

original_rows = len(df)
original_columns = len(df.columns)

if df["SEQN"].isna().any() or not df["SEQN"].is_unique:
    raise ValueError("Invalid participant identifiers")

print("Original participants:", original_rows)
print("Original columns:", original_columns)

print("\nStandardizing laboratory features...")

for new_name, source_column in FEATURE_MAP.items():
    if source_column not in df.columns:
        raise KeyError(f"Missing source: {source_column}")

    if new_name in df.columns:
        raise ValueError(f"Feature already exists: {new_name}")

    df[new_name] = pd.to_numeric(
        df[source_column], errors="coerce"
    )

    print(
        f"{new_name}: "
        f"{df[new_name].notna().sum()} available"
    )


# Blood pressure:
# Average valid observed readings, without imputing
# missing readings or interpreting clinical risk.
for kind, prefix in [
    ("systolic", "BPXSY"),
    ("diastolic", "BPXDI"),
]:
    columns = [
        f"BPX__{prefix}{i}" for i in range(1, 5)
    ]

    for column in columns:
        if column not in df.columns:
            raise KeyError(f"Missing BP column: {column}")

    readings = df[columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # Remove nonphysiological values.
    # These are quality filters, not clinical thresholds.
    if kind == "systolic":
        readings = readings.where(
            (readings > 0) & (readings <= 300)
        )
    else:
        readings = readings.where(
            (readings > 0) & (readings <= 200)
        )

    mean_name = f"{kind}_bp_mean_mmhg"
    count_name = f"{kind}_bp_reading_count"

    df[mean_name] = readings.mean(axis=1)
    df[count_name] = readings.notna().sum(axis=1)

    print(
        f"{mean_name}: "
        f"{df[mean_name].notna().sum()} available"
    )


print("\nModule feature coverage:")

coverage = {}

for module, features in MODULE_FEATURES.items():
    missing_columns = [
        col for col in features if col not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"{module} missing columns: {missing_columns}"
        )

    complete = int(
        df[features].notna().all(axis=1).sum()
    )

    coverage[module] = {
        "features": features,
        "participants_with_all_listed_features": complete,
    }

    print(f"{module}: {complete}")


if len(df) != original_rows or not df["SEQN"].is_unique:
    raise RuntimeError("Dataset integrity check failed")


feature_names = list(FEATURE_MAP) + [
    "systolic_bp_mean_mmhg",
    "diastolic_bp_mean_mmhg",
    "systolic_bp_reading_count",
    "diastolic_bp_reading_count",
]

report = {
    "source_dataset": INPUT.name,
    "output_dataset": OUTPUT.name,
    "population": "NHANES 2017-2018 research participants",
    "rows": int(len(df)),
    "columns": int(len(df.columns)),
    "unique_participants": int(df["SEQN"].nunique()),
    "new_features": {
        name: {
            "non_missing": int(df[name].notna().sum()),
            "missing": int(df[name].isna().sum()),
        }
        for name in feature_names
    },
    "module_feature_coverage": coverage,
    "limitations": [
        "Module coverage is not diagnostic readiness.",
        "Kidney stone inputs do not establish stone risk.",
        "Blood pressure averages are descriptive only.",
        "Source units require documentation verification.",
        "No astronaut-specific or longitudinal outcomes.",
    ],
}

df.to_csv(OUTPUT, index=False)

with REPORT.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 65)
print("FEATURE DATASET V3 CREATED")
print("=" * 65)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Unique participants:", df["SEQN"].nunique())
print("Saved:", OUTPUT)
print("Quality report:", REPORT)
