
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ("DEMO", "DEMO_J.xpt"),
    ("BMX", "BMX_J.xpt"),
    ("DR1", "DR1TOT_J.xpt"),
    ("DR2", "DR2TOT_J.xpt"),
    ("CBC", "CBC_J.XPT"),
    ("VID", "VID_J.XPT"),
    ("WHQ", "WHQ_J.xpt"),
    ("DS1", "DS1TOT_J.XPT"),
    ("DSQ", "DSQTOT_J.XPT"),
]

# Standardized names map to (source, original NHANES column).
FEATURES = {
    "age_years": ("DEMO", "RIDAGEYR"),
    "sex_code": ("DEMO", "RIAGENDR"),
    "weight_kg": ("BMX", "BMXWT"),
    "height_cm": ("BMX", "BMXHT"),
    "bmi": ("BMX", "BMXBMI"),
    "waist_cm": ("BMX", "BMXWAIST"),
    "hip_cm": ("BMX", "BMXHIP"),
    "hemoglobin_g_dl": ("CBC", "LBXHGB"),
    "hematocrit_pct": ("CBC", "LBXHCT"),
    "rbc_count": ("CBC", "LBXRBCSI"),
    "mcv_fl": ("CBC", "LBXMCVSI"),
    "mch_pg": ("CBC", "LBXMCHSI"),
    "rdw_pct": ("CBC", "LBXRDW"),
    "serum_vitamin_d_nmol_l": ("VID", "LBXVIDMS"),
}

# Only standardize dietary fields for which source units
# and meanings are understood.
DIETARY_FIELDS = {
    "energy_kcal": "TKCAL",
    "protein_g": "TPROT",
    "carbohydrate_g": "TCARB",
    "fat_g": "TTFAT",
    "fiber_g": "TFIBE",
    "calcium_mg": "TCALC",
    "iron_mg": "TIRON",
    "vitamin_d_mcg": "TVD",
    "vitamin_b12_mcg": "TVB12",
    "vitamin_c_mg": "TVC",
    "magnesium_mg": "TMAGN",
    "zinc_mg": "TZINC",
    "potassium_mg": "TPOTA",
    "sodium_mg": "TSODI",
    "phosphorus_mg": "TPHOS",
    "selenium_mcg": "TSELE",
    "vitamin_b6_mg": "TVB6",
    "vitamin_k_mcg": "TVK",
    "folate_dfe_mcg": "TFDFE",
}

for day in (1, 2):
    for name, suffix in DIETARY_FIELDS.items():
        FEATURES[f"day{day}_{name}"] = (
            f"DR{day}",
            f"DR{day}{suffix}",
        )


def read_source(source_name, filename):
    path = RAW / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    frame = pd.read_sas(path, format="xport")

    if "SEQN" not in frame.columns:
        raise ValueError(f"{filename} has no SEQN column")

    if frame["SEQN"].isna().any():
        raise ValueError(f"{filename} contains missing SEQN")

    if frame["SEQN"].duplicated().any():
        raise ValueError(f"{filename} contains duplicate SEQN")

    # Prefix original fields to prevent collisions across files.
    # This preserves every source variable independently.
    frame = frame.rename(
        columns={
            column: f"{source_name}__{column}"
            for column in frame.columns
            if column != "SEQN"
        }
    )

    return frame


print("Loading datasets...")

datasets = {}
source_report = {}

for source_name, filename in SOURCES:
    frame = read_source(source_name, filename)
    datasets[source_name] = frame

    source_report[source_name] = {
        "filename": filename,
        "rows": len(frame),
        "columns": len(frame.columns),
    }

    print(f"{source_name}: {len(frame)} participants")


print("\nMerging datasets...")

master = datasets["DEMO"].copy()

for source_name, _ in SOURCES[1:]:
    before = len(master)

    master = master.merge(
        datasets[source_name],
        on="SEQN",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if len(master) != before:
        raise RuntimeError(
            f"Unexpected row-count change after {source_name}"
        )

    print(f"Merged {source_name}: {len(master)} rows")


print("\nCreating standardized features...")

for new_name, (source_name, original_name) in FEATURES.items():
    source_column = f"{source_name}__{original_name}"

    if source_column not in master.columns:
        raise KeyError(
            f"Required feature missing: {source_column}"
        )

    master[new_name] = pd.to_numeric(
        master[source_column],
        errors="coerce",
    )


# Dietary averages: use only reported days.
# This is an observed two-day average, not habitual intake.
for nutrient in DIETARY_FIELDS:
    first = f"day1_{nutrient}"
    second = f"day2_{nutrient}"

    master[f"avg_{nutrient}"] = master[
        [first, second]
    ].mean(axis=1, skipna=True)


# Source coverage is different from measurement availability.
for source_name, _ in SOURCES[1:]:
    source_ids = datasets[source_name]["SEQN"]
    master[f"has_{source_name.lower()}_record"] = (
        master["SEQN"].isin(source_ids)
    )


output_csv = PROCESSED / "astronaut_nutrition_master.csv"

master.to_csv(output_csv, index=False)


report = {
    "dataset": output_csv.name,
    "population": "NHANES 2017-2018 research participants",
    "rows": int(len(master)),
    "columns": int(len(master.columns)),
    "unique_participants": int(master["SEQN"].nunique()),
    "sources": source_report,
    "standardized_features": {
        name: {
            "source": source,
            "original_column": original,
            "non_missing": int(master[name].notna().sum()),
            "missing": int(master[name].isna().sum()),
        }
        for name, (source, original) in FEATURES.items()
    },
    "all_column_missing_counts": {
        column: int(count)
        for column, count in master.isna().sum().items()
    },
}

output_json = REPORTS / "master_dataset_quality.json"

with output_json.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)


print("\n" + "=" * 60)
print("MASTER DATASET CREATED")
print("=" * 60)
print("Rows:", len(master))
print("Columns:", len(master.columns))
print("Unique participants:", master["SEQN"].nunique())
print("CSV:", output_csv)
print("Quality report:", output_json)

print("\nKey feature availability:")

for name in [
    "age_years",
    "bmi",
    "avg_protein_g",
    "avg_iron_mg",
    "hemoglobin_g_dl",
    "serum_vitamin_d_nmol_l",
]:
    print(f"{name}: {master[name].notna().sum()}")

