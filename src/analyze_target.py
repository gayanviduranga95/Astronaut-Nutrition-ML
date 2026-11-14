
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed" / "nutrition_merged.csv"

df = pd.read_csv(DATA)

print("\n=== DATASET ===")
print("Total participants:", len(df))

# Examine weight-change distributions.
change = df["weight_change_pct"]

print("\n=== WEIGHT CHANGE ===")
print(change.describe(
    percentiles=[0.01, 0.05, 0.10, 0.90, 0.95, 0.99]
))

# Exploratory weight-loss indicators.
# These are NOT clinical malnutrition labels.
df["weight_loss_5pct"] = (
    change <= -5
).astype("Int64")

df.loc[
    change.isna(), "weight_loss_5pct"
] = pd.NA

df["weight_loss_10pct"] = (
    change <= -10
).astype("Int64")

df.loc[
    change.isna(), "weight_loss_10pct"
] = pd.NA

for col in ["weight_loss_5pct", "weight_loss_10pct"]:
    print(f"\n=== {col} ===")
    print(df[col].value_counts(dropna=False))
    print(
        "Percentage among known records:",
        round(df[col].mean() * 100, 2)
    )

print("\n=== REPORTED WEIGHT-LOSS INTENT ===")
print(
    df["tried_to_lose_weight"]
    .value_counts(dropna=False)
)

print("\n=== DIETARY INTAKE BY WEIGHT-LOSS GROUP ===")
columns = [
    "avg_energy_kcal",
    "avg_protein_g",
    "avg_calcium_mg",
    "avg_iron_mg",
    "avg_vitamin_d_mcg",
    "bmi"
]

print(
    df.groupby("weight_loss_5pct")[columns]
    .median()
    .round(2)
    .to_string()
)

print("\n=== MISSING VALUES ===")
print(df[columns + ["weight_change_pct"]].isna().sum())

print("\nAnalysis completed.")
