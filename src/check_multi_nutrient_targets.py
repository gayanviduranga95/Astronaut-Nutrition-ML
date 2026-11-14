
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA_PATH = (
    BASE / "data" / "processed"
    / "nutrition_with_labs.csv"
)

df = pd.read_csv(DATA_PATH)

lab_columns = [
    "LBXVIDMS",
    "LBXHGB",
    "LBXRBCSI",
    "LBXMCVSI",
]

print("=" * 60)
print("ASTRONAUT NUTRITION ML - TARGET EXPLORATION")
print("=" * 60)

for column in lab_columns:
    values = pd.to_numeric(df[column], errors="coerce")

    print(f"\n{column}")
    print("-" * 40)
    print("Available:", values.notna().sum())
    print("Missing:", values.isna().sum())
    print("Mean:", round(values.mean(), 3))
    print("Median:", round(values.median(), 3))
    print("Minimum:", round(values.min(), 3))
    print("Maximum:", round(values.max(), 3))

    print("\nPercentiles:")
    print(
        values.quantile(
            [0.05, 0.25, 0.50, 0.75, 0.95]
        ).round(3).to_string()
    )

print("\nHEMOGLOBIN BY SEX CODE")
print(
    df.groupby("sex_code")["LBXHGB"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(3)
    .to_string()
)

print("\nHEMOGLOBIN AND IRON INTAKE")
paired = df[
    ["avg_iron_mg", "LBXHGB"]
].dropna()

print(
    "Correlation:",
    round(
        paired["avg_iron_mg"].corr(
            paired["LBXHGB"]
        ),
        4,
    ),
)

print("\nEXPLORATION COMPLETE")
print(
    "These results do not establish "
    "iron deficiency or astronaut-specific risk."
)
multi_nutrient_engine.py