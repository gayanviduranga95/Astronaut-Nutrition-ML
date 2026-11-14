
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"

files = {
    "Dietary Day 1": "DR1TOT_J.xpt",
    "Dietary Day 2": "DR2TOT_J.xpt",
    "Weight History": "WHQ_J.xpt",
}

for name, filename in files.items():
    df = pd.read_sas(RAW / filename, format="xport")

    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")

    print("\nAll columns:")
    for column in df.columns:
        print(column)

print("\nInspection complete!")
