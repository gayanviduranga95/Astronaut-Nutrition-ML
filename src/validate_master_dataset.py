
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

df = pd.read_csv(
    DATA / "astronaut_nutrition_master.csv",
    low_memory=False,
)

print("=" * 60)
print("STEP 35: MASTER DATASET VALIDATION")
print("=" * 60)

checks = {}

checks["unique_participants"] = bool(
    df["SEQN"].is_unique
    and df["SEQN"].notna().all()
)

checks["expected_row_count"] = len(df) == 9254

ranges = {
    "age_years": (0, 120),
    "bmi": (10, 100),
    "weight_kg": (2, 400),
    "height_cm": (40, 250),
    "hemoglobin_g_dl": (3, 25),
    "serum_vitamin_d_nmol_l": (0, 500),
}

print("\nMeasurement screening:")
for name, (low, high) in ranges.items():
    values = pd.to_numeric(df[name], errors="coerce")
    invalid = (
        values.notna()
        & ~values.between(low, high)
    ).sum()

    print(
        f"{name}: available={values.notna().sum()}, "
        f"outside_screening_range={invalid}"
    )

    checks[f"{name}_range_screen"] = int(invalid) == 0

# Verify that averages match the available recall days.
for nutrient in ["protein_g", "iron_mg", "calcium_mg"]:
    day1 = f"day1_{nutrient}"
    day2 = f"day2_{nutrient}"
    average = f"avg_{nutrient}"

    expected = df[[day1, day2]].mean(
        axis=1, skipna=True
    )

    actual = df[average]

    correct = np.isclose(
        expected.to_numpy(dtype=float),
        actual.to_numpy(dtype=float),
        rtol=1e-8,
        atol=1e-8,
        equal_nan=True,
    ).all()

    checks[f"{nutrient}_average"] = bool(correct)

# Data availability only: not clinical model readiness.
modules = {
    "nutritional_status": [
        "avg_protein_g", "avg_iron_mg", "bmi"
    ],
    "anemia_iron": [
        "hemoglobin_g_dl", "mcv_fl", "avg_iron_mg"
    ],
    "glucose_regulation": [
        "bmi"
    ],
    "kidney_health": [],
    "kidney_stone": [],
    "liver_health": [],
    "cardiovascular_metabolic": [
        "bmi", "waist_cm"
    ],
    "hydration_electrolytes": [
        "avg_sodium_mg", "avg_potassium_mg"
    ],
    "bone_nutrition": [
        "serum_vitamin_d_nmol_l",
        "avg_calcium_mg",
        "avg_phosphorus_mg"
    ],
}

coverage = {}

print("\nModule input availability:")

for module, features in modules.items():
    if features:
        count = int(df[features].notna().all(axis=1).sum())
    else:
        count = 0

    coverage[module] = {
        "available_input_features": features,
        "participants_with_all_listed_features": count,
        "additional_features_required": True,
    }

    print(f"{module}: {count} participants")

report = {
    "rows": len(df),
    "columns": len(df.columns),
    "checks": checks,
    "module_input_coverage": coverage,
    "note": (
        "Screening ranges are data-quality heuristics, "
        "not clinical reference intervals. Module coverage "
        "does not establish diagnostic or ML readiness."
    ),
}

REPORTS.mkdir(parents=True, exist_ok=True)

output = REPORTS / "master_validation.json"

with output.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\nValidation checks:")
for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'REVIEW'}")

print("\nSaved:", output)
