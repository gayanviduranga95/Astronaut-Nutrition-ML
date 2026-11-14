
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE / "data" / "processed" / "nutrition_merged.csv"
)

# Weight-loss indicator: exploratory only
df = df.dropna(subset=["weight_change_pct"]).copy()
df["weight_loss_5pct"] = df["weight_change_pct"] <= -5

print("\n=== WEIGHT LOSS AND REPORTED INTENT ===")

print(pd.crosstab(
    df["weight_loss_5pct"],
    df["tried_to_lose_weight"],
    margins=True,
    dropna=False
))

print("\n=== PERCENTAGES WITHIN EACH GROUP ===")

print(
    pd.crosstab(
        df["weight_loss_5pct"],
        df["tried_to_lose_weight"],
        normalize="index",
        dropna=False
    ).round(3) * 100
)

print("\n=== MISSING INTENT INFORMATION ===")
print(df["tried_to_lose_weight"].isna().sum())
