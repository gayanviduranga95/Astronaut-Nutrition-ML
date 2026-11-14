
from pathlib import Path
import json
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
REPORTS = BASE / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)

inventory = {}

for path in sorted(RAW.iterdir()):
    if path.suffix.lower() != ".xpt":
        continue

    print(f"\n{'=' * 60}")
    print(f"DATASET: {path.name}")
    print("=" * 60)

    df = pd.read_sas(path, format="xport")

    print("Participants/rows:", len(df))
    print("Columns:", len(df.columns))

    columns = []

    for column in df.columns:
        available = int(df[column].notna().sum())

        columns.append({
            "name": column,
            "available": available,
            "missing": int(df[column].isna().sum()),
        })

        print(f"{column}: {available} available")

    inventory[path.name] = {
        "rows": len(df),
        "column_count": len(df.columns),
        "columns": columns,
    }

output = REPORTS / "dataset_inventory.json"

with open(output, "w", encoding="utf-8") as file:
    json.dump(inventory, file, indent=2)

print("\n" + "=" * 60)
print("DATASET AUDIT COMPLETE")
print("Saved report:", output)
print("=" * 60)

