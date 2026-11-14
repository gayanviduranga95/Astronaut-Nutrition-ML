
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"

nutrition = pd.read_csv(
    PROCESSED / "nutrition_merged.csv"
)

vitamin_d = pd.read_sas(
    RAW / "VID_J.xpt",
    format="xport"
)[["SEQN", "LBXVIDMS"]]

cbc = pd.read_sas(
    RAW / "CBC_J.xpt",
    format="xport"
)[["SEQN", "LBXHGB", "LBXRBCSI", "LBXMCVSI"]]

df = nutrition.merge(
    vitamin_d,
    on="SEQN",
    how="left",
    validate="one_to_one"
)

df = df.merge(
    cbc,
    on="SEQN",
    how="left",
    validate="one_to_one"
)

print("\n========== LAB DATA SUMMARY ==========")
print("Total nutrition records:", len(df))

for column in [
    "LBXVIDMS",
    "LBXHGB",
    "LBXRBCSI",
    "LBXMCVSI"
]:
    print(f"\n{column}")
    print("Available:", df[column].notna().sum())
    print("Missing:", df[column].isna().sum())
    print(df[column].describe().round(2))

# Exploratory vitamin D status categories.
# These are not astronaut-specific clinical thresholds.
vitamin_d_valid = df["LBXVIDMS"].dropna()

print("\n========== VITAMIN D DISTRIBUTION ==========")

print("Below 30 nmol/L:",
      (vitamin_d_valid < 30).sum())

print("30 to below 50 nmol/L:",
      ((vitamin_d_valid >= 30) &
       (vitamin_d_valid < 50)).sum())

print("50 nmol/L or above:",
      (vitamin_d_valid >= 50).sum())

print("\n========== OVERLAP ==========")

print(
    "Records with both vitamin D and hemoglobin:",
    df[["LBXVIDMS", "LBXHGB"]]
    .notna()
    .all(axis=1)
    .sum()
)

output = PROCESSED / "nutrition_with_labs.csv"
df.to_csv(output, index=False)

print("\nSaved:", output)
print("SUCCESS!")
