
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"

# Existing enhanced dataset
nutrition = pd.read_csv(
    PROCESSED / "nutrition_enhanced_ml.csv"
)

# 30-day dietary supplement dataset
supplements = pd.read_sas(
    RAW / "DSQTOT_J.xpt",
    format="xport"
)

columns = [
    "SEQN",
    "DSQTVD",
    "DSDCOUNT",
    "DSD010",
]

supplements = supplements[columns].copy()

supplements = supplements.rename(columns={
    "DSQTVD": "vitamin_d_supplement_30day_mcg",
    "DSDCOUNT": "supplement_count_30day",
    "DSD010": "supplement_use_30day",
})

# Merge without losing existing participants
df = nutrition.merge(
    supplements,
    on="SEQN",
    how="left",
    validate="one_to_one",
)

for column in [
    "vitamin_d_supplement_30day_mcg",
    "supplement_count_30day",
    "supplement_use_30day",
]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

# SAS XPT sometimes represents zero using
# extremely small floating-point values.
for column in [
    "vitamin_d_supplement_30day_mcg",
    "supplement_count_30day",
]:
    df.loc[
        df[column].abs() < 1e-20,
        column
    ] = 0

# Do not assume other missing values are zero.
df["vitamin_d_30day_missing"] = (
    df["vitamin_d_supplement_30day_mcg"]
    .isna()
    .astype(int)
)

output = PROCESSED / "nutrition_30day_ml.csv"

df.to_csv(output, index=False)

print("\n30-DAY DATASET CREATED")
print("Records:", len(df))
print("Columns:", len(df.columns))

print("\nNEW FEATURE AVAILABILITY")
print(
    df[
        [
            "vitamin_d_supplement_30day_mcg",
            "supplement_count_30day",
            "supplement_use_30day",
        ]
    ].notna().sum()
)

print("\n30-DAY SUPPLEMENT COUNT DISTRIBUTION")
print(
    df["supplement_count_30day"]
    .value_counts(dropna=False)
    .sort_index()
    .head(20)
)

print("\nVITAMIN D SUPPLEMENT INTAKE")
print(
    df["vitamin_d_supplement_30day_mcg"]
    .describe()
)

print("\nMISSING VITAMIN D BY SUPPLEMENT COUNT")
print(
    pd.crosstab(
        df["supplement_count_30day"].fillna(-1),
        df["vitamin_d_30day_missing"],
        margins=True
    ).head(20)
)

print("\nSaved:", output)

