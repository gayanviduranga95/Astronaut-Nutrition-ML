
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

path = BASE / "data" / "raw" / "DSQTOT_J.xpt"

if not path.exists():
    raise FileNotFoundError(
        f"Download DSQTOT_J.xpt first: {path}"
    )

df = pd.read_sas(path, format="xport")

print("\n30-DAY SUPPLEMENT DATASET")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nALL COLUMNS")
for column in df.columns:
    print(column)

print("\nVITAMIN D AND SUPPLEMENT VARIABLES")
selected = [
    col for col in df.columns
    if "VD" in col.upper()
    or "DSQ" in col.upper()
]

print(selected)

print("\nINSPECTION COMPLETE")
