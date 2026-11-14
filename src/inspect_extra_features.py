
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"

files = {
    "Demographics": "DEMO_J.xpt",
    "Dietary Supplements (30-day total)": "DS1TOT_J.xpt",
}

for title, filename in files.items():
    path = RAW / filename

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    if not path.exists():
        print(f"File not found: {filename}")
        print("Download or verify the filename first.")
        continue

    df = pd.read_sas(path, format="xport")

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    for column in df.columns:
        print(column)

print("\nINSPECTION COMPLETE")
