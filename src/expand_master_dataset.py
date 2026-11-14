
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

INPUT_FILE = (
    PROCESSED / "astronaut_nutrition_master.csv"
)

OUTPUT_FILE = (
    PROCESSED / "astronaut_nutrition_master_v2.csv"
)

REPORT_FILE = (
    REPORTS / "master_v2_quality.json"
)

NEW_SOURCES = {
    "BIOPRO": "BIOPRO_J.xpt",
    "GHB": "GHB_J.xpt",
    "FERTIN": "FERTIN_J.xpt",
    "HDL": "HDL_J.xpt",
    "TCHOL": "TCHOL_J.xpt",
    "TRIGLY": "TRIGLY_J.xpt",
    "ALB_CR": "ALB_CR_J.xpt",
    "BPX": "BPX_J.xpt",
}


def load_source(name, filename):
    path = RAW / filename

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_sas(path, format="xport")

    if "SEQN" not in df.columns:
        raise ValueError(
            f"{filename} is missing SEQN"
        )

    if df["SEQN"].isna().any():
        raise ValueError(
            f"{filename} has missing SEQN"
        )

    if df["SEQN"].duplicated().any():
        raise ValueError(
            f"{filename} has duplicate SEQN"
        )

    df = df.rename(
        columns={
            col: f"{name}__{col}"
            for col in df.columns
            if col != "SEQN"
        }
    )

    return df


def main():
    print("=" * 60)
    print("STEP 37: EXPAND MASTER DATASET")
    print("=" * 60)

    master = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    original_rows = len(master)

    if master["SEQN"].duplicated().any():
        raise ValueError(
            "Master dataset contains duplicate SEQN"
        )

    if master["SEQN"].isna().any():
        raise ValueError(
            "Master dataset contains missing SEQN"
        )

    print("Original rows:", original_rows)
    print("Original columns:", len(master.columns))

    source_report = {}

    for name, filename in NEW_SOURCES.items():
        print(f"\nProcessing {name}...")

        df = load_source(name, filename)

        overlap = (
            set(master.columns)
            & (set(df.columns) - {"SEQN"})
        )

        if overlap:
            raise ValueError(
                f"Unexpected duplicate columns: {overlap}"
            )

        matched = int(
            master["SEQN"].isin(df["SEQN"]).sum()
        )

        master = master.merge(
            df,
            on="SEQN",
            how="left",
            validate="one_to_one",
            sort=False,
        )

        if len(master) != original_rows:
            raise RuntimeError(
                f"Row count changed after {name}"
            )

        source_report[name] = {
            "filename": filename,
            "source_rows": len(df),
            "source_columns": len(df.columns),
            "matched_participants": matched,
        }

        print("Source rows:", len(df))
        print("Matched participants:", matched)
        print("Current columns:", len(master.columns))

    if not master["SEQN"].is_unique:
        raise RuntimeError(
            "Duplicate participant IDs after merge"
        )

    report = {
        "population": (
            "NHANES 2017-2018 Earth-population "
            "research dataset"
        ),
        "rows": int(len(master)),
        "columns": int(len(master.columns)),
        "unique_participants": int(
            master["SEQN"].nunique()
        ),
        "new_sources": source_report,
        "new_lab_columns": [
            col for col in master.columns
            if any(
                col.startswith(f"{name}__")
                for name in NEW_SOURCES
            )
        ],
    }

    master.to_csv(OUTPUT_FILE, index=False)

    with REPORT_FILE.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 60)
    print("MASTER DATASET V2 CREATED")
    print("=" * 60)

    print("Rows:", len(master))
    print("Columns:", len(master.columns))
    print(
        "Unique participants:",
        master["SEQN"].nunique()
    )
    print("Saved:", OUTPUT_FILE)
    print("Report:", REPORT_FILE)


if __name__ == "__main__":
    main()
