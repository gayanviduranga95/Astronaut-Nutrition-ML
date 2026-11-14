
from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATASETS
# ==========================================

def load_dataset(filename, columns):
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    print(f"Loading {filename}...")

    df = pd.read_sas(path, format="xport")

    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{filename} is missing columns: {missing}"
        )

    return df[columns].copy()


# Demographic information
demo = load_dataset("DEMO_J.xpt", [
    "SEQN",
    "RIDAGEYR",
    "RIAGENDR"
])

# Body measurements
bmx = load_dataset("BMX_J.xpt", [
    "SEQN",
    "BMXWT",
    "BMXHT",
    "BMXBMI"
])


# ==========================================
# 3. DIETARY NUTRIENTS
# ==========================================

# NHANES nutrient variable suffixes.
# These are present in both Day 1 and Day 2.
nutrient_variables = {
    "TKCAL": "energy_kcal",
    "TPROT": "protein_g",
    "TCARB": "carbohydrate_g",
    "TTFAT": "fat_g",
    "TCALC": "calcium_mg",
    "TIRON": "iron_mg",
    "TVD": "vitamin_d_mcg",
    "TFIBE": "fiber_g",
    "TVB12": "vitamin_b12_mcg",
    "TVC": "vitamin_c_mg",
    "TMAGN": "magnesium_mg",
    "TZINC": "zinc_mg",
    "TPOTA": "potassium_mg"
}

day1_columns = [
    "SEQN",
    "DR1DRSTZ"
] + [
    f"DR1{suffix}"
    for suffix in nutrient_variables
]

day2_columns = [
    "SEQN",
    "DR2DRSTZ"
] + [
    f"DR2{suffix}"
    for suffix in nutrient_variables
]

day1 = load_dataset(
    "DR1TOT_J.xpt",
    day1_columns
)

day2 = load_dataset(
    "DR2TOT_J.xpt",
    day2_columns
)


# ==========================================
# 4. WEIGHT HISTORY
# ==========================================

whq = load_dataset("WHQ_J.xpt", [
    "SEQN",
    "WHD020",
    "WHD050",
    "WHQ070"
])


# ==========================================
# 5. MERGE DATASETS
# ==========================================

print("\nMerging datasets...")

df = demo.merge(
    bmx,
    on="SEQN",
    how="left",
    validate="one_to_one"
)

df = df.merge(
    day1,
    on="SEQN",
    how="left",
    validate="one_to_one"
)

df = df.merge(
    day2,
    on="SEQN",
    how="left",
    validate="one_to_one"
)

df = df.merge(
    whq,
    on="SEQN",
    how="left",
    validate="one_to_one"
)

print("Total merged participants:", len(df))


# ==========================================
# 6. FILTER ADULT PARTICIPANTS
# ==========================================

print("\nFiltering adult participants...")

df = df[
    (df["RIDAGEYR"] >= 18) &
    (df["DR1DRSTZ"] == 1) &
    (df["DR2DRSTZ"] == 1)
].copy()

print("Participants after filtering:", len(df))


# ==========================================
# 7. CLEAN INVALID VALUES
# ==========================================

print("\nCleaning invalid values...")

# Weight history values are reported in pounds.
# Remove special nonresponse codes.

for col in ["WHD020", "WHD050"]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    df.loc[
        (df[col] <= 0) |
        (df[col] >= 7777),
        col
    ] = np.nan


# Clean body measurements.
for col in ["BMXWT", "BMXHT", "BMXBMI"]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    df.loc[df[col] <= 0, col] = np.nan


# Convert nutrient columns to numeric.
for day in [1, 2]:
    for suffix in nutrient_variables:

        col = f"DR{day}{suffix}"

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        df.loc[df[col] < 0, col] = np.nan


# ==========================================
# 8. CALCULATE WEIGHT CHANGE
# ==========================================

# Negative = reported weight loss
# Positive = reported weight gain
#
# This is exploratory information, not
# a clinical malnutrition diagnosis.

df["weight_change_pct"] = (
    (df["WHD020"] - df["WHD050"])
    / df["WHD050"]
) * 100


# ==========================================
# 9. RENAME COLUMNS
# ==========================================

rename_map = {
    "RIDAGEYR": "age_years",
    "RIAGENDR": "sex_code",
    "BMXWT": "weight_kg",
    "BMXHT": "height_cm",
    "BMXBMI": "bmi",
    "WHQ070": "tried_to_lose_weight"
}

# Rename Day 1 and Day 2 nutrient columns.
for suffix, nutrient_name in nutrient_variables.items():

    rename_map[f"DR1{suffix}"] = (
        f"day1_{nutrient_name}"
    )

    rename_map[f"DR2{suffix}"] = (
        f"day2_{nutrient_name}"
    )

df = df.rename(columns=rename_map)


# ==========================================
# 10. CALCULATE AVERAGE NUTRIENT INTAKE
# ==========================================

print("\nCalculating average nutrient intake...")

for nutrient in nutrient_variables.values():

    day1_col = f"day1_{nutrient}"
    day2_col = f"day2_{nutrient}"

    df[f"avg_{nutrient}"] = df[
        [day1_col, day2_col]
    ].mean(axis=1)


# ==========================================
# 11. SAVE PROCESSED DATA
# ==========================================

output_file = (
    OUTPUT_DIR / "nutrition_merged.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ==========================================
# 12. DATASET REPORT
# ==========================================

print("\n====================================")
print("DATA PREPARATION COMPLETED")
print("====================================")

print("Total participants:", len(df))
print("Total columns:", len(df.columns))

print("\nMissing values:")
print(
    df.isna()
    .sum()
    .sort_values(ascending=False)
    .head(20)
)

print("\nWeight change statistics:")
print(
    df["weight_change_pct"].describe()
)

average_columns = [
    f"avg_{nutrient}"
    for nutrient in nutrient_variables.values()
]

print("\nAverage nutrient intake statistics:")
print(
    df[average_columns]
    .describe()
    .round(2)
    .to_string()
)

print("\nProcessed dataset saved to:")
print(output_file)

print("\nSUCCESS!")
