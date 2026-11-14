
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"

nutrition = pd.read_csv(
    PROCESSED / "nutrition_with_labs.csv"
)

demo = pd.read_sas(
    RAW / "DEMO_J.xpt",
    format="xport"
)[["SEQN", "RIDEXMON"]]

supplements = pd.read_sas(
    RAW / "DS1TOT_J.xpt",
    format="xport"
)[["SEQN", "DS1TVD", "DS1DSCNT"]]

# Preserve all existing nutrition records.
df = nutrition.merge(
    demo,
    on="SEQN",
    how="left",
    validate="one_to_one"
)

df = df.merge(
    supplements,
    on="SEQN",
    how="left",
    validate="one_to_one"
)

df = df.rename(columns={
    "RIDEXMON": "examination_period",
    "DS1TVD": "supplement_vitamin_d_mcg",
    "DS1DSCNT": "supplement_count"
})

# Preserve missing supplement values rather than
# automatically assuming that missing means zero.
for column in [
    "supplement_vitamin_d_mcg",
    "supplement_count"
]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    df.loc[df[column] < 0, column] = np.nan

# Examination period is categorical.
df["examination_period"] = pd.to_numeric(
    df["examination_period"],
    errors="coerce"
)

output = PROCESSED / "nutrition_enhanced_ml.csv"
df.to_csv(output, index=False)

print("\nENHANCED DATASET CREATED")
print("Records:", len(df))
print("Columns:", len(df.columns))

print("\nNew feature availability:")
print(
    df[
        [
            "supplement_vitamin_d_mcg",
            "supplement_count",
            "examination_period"
        ]
    ].notna().sum()
)

print("\nSupplement vitamin D distribution:")
print(
    df["supplement_vitamin_d_mcg"]
    .describe()
    .round(2)
)

print("\nExamination period distribution:")
print(
    df["examination_period"]
    .value_counts(dropna=False)
)

print("\nSaved:", output)
