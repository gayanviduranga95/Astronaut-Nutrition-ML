
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data" / "processed" / "nutrition_with_labs.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("ASTRONAUT NUTRITION ML - DATA INSPECTION")
print("=" * 60)

print("\nTotal participants:", len(df))
print("Total columns:", len(df.columns))

print("\n1. AVAILABLE COLUMNS")
for column in df.columns:
    print(column)

print("\n2. NUTRITION FEATURES")

nutrition_columns = [
    col for col in df.columns
    if col.startswith("avg_")
]

if nutrition_columns:
    summary = df[nutrition_columns].describe().T
    summary["missing_count"] = df[nutrition_columns].isna().sum()
    summary["missing_percent"] = (
        df[nutrition_columns].isna().mean() * 100
    ).round(2)

    print(summary[
        ["count", "mean", "min", "max", "missing_percent"]
    ].to_string())
else:
    print("No average nutrient columns found.")

print("\n3. POTENTIAL LABORATORY TARGETS")

lab_keywords = [
    "LBX", "LBD", "hemoglobin",
    "ferritin", "vitamin", "iron"
]

lab_columns = [
    col for col in df.columns
    if any(
        keyword.lower() in col.lower()
        for keyword in lab_keywords
    )
]

if lab_columns:
    for col in lab_columns:
        print(
            f"{col}: "
            f"{df[col].notna().sum()} available, "
            f"{df[col].isna().sum()} missing"
        )
else:
    print("No potential laboratory columns found.")

print("\n4. DATA QUALITY")

print("Duplicate rows:", df.duplicated().sum())

if "SEQN" in df.columns:
    print(
        "Duplicate participant IDs:",
        df["SEQN"].duplicated().sum()
    )

print("\n5. NEXT MODEL OPPORTUNITIES")

print("Nutrition intake assessment: available")
print("Vitamin D laboratory target: previously explored")
print("Other laboratory targets: require verification")
print("Long-term astronaut forecasts: need time-series data")

print("\nINSPECTION COMPLETE")
