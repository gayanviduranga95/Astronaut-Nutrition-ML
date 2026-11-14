
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE / "data" / "processed"
    / "nutrition_enhanced_ml.csv"
)

df = df.dropna(subset=["LBXVIDMS"]).copy()

df["low_vitamin_d"] = (
    df["LBXVIDMS"] < 50
).astype(int)

df["vitamin_d_amount_missing"] = (
    df["supplement_vitamin_d_mcg"].isna()
)

print("\nSUPPLEMENT COUNT DISTRIBUTION")
print(
    df["supplement_count"]
    .value_counts(dropna=False)
    .sort_index()
    .head(20)
)

print("\nMISSING VITAMIN D AMOUNT BY SUPPLEMENT COUNT")
print(
    pd.crosstab(
        df["supplement_count"].fillna(-1),
        df["vitamin_d_amount_missing"],
        margins=True
    ).head(25)
)

print("\nLOW VITAMIN D RATE BY MISSINGNESS")
print(
    df.groupby("vitamin_d_amount_missing")[
        "low_vitamin_d"
    ].agg(["count", "mean"])
)

print("\nLOW VITAMIN D RATE BY SUPPLEMENT COUNT")
print(
    df.groupby("supplement_count")[
        "low_vitamin_d"
    ].agg(["count", "mean"])
    .head(20)
)

print("\nCHECK COMPLETE")
