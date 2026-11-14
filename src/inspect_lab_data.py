
from pathlib import Path
import pandas as pd

raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"

for filename in ["VID_J.xpt", "CBC_J.xpt"]:
    path = raw_dir / filename

    if not path.exists():
        print(f"Missing: {filename}")
        continue

    df = pd.read_sas(path, format="xport")

    print(f"\n{'=' * 50}")
    print(filename)
    print(f"Rows: {len(df)}")
    print(f"{'=' * 50}")

    for column in df.columns:
        print(column)
