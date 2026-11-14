
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

SOURCE = DATA / "astronaut_nutrition_master_v2.csv"
FEATURES = DATA / "astronaut_nutrition_features_v3.csv"
OUTPUT = REPORTS / "features_v3_validation.json"

source = pd.read_csv(SOURCE, low_memory=False)
df = pd.read_csv(FEATURES, low_memory=False)

checks = {}

def check(name, condition):
    passed = bool(condition)
    checks[name] = passed
    print(f"{'PASS' if passed else 'FAIL'}: {name}")

print("=" * 60)
print("STEP 40: FEATURE DATASET VALIDATION")
print("=" * 60)

check("Row count preserved", len(source) == len(df))
check("Expected participants", len(df) == 9254)
check("Unique participant IDs", df["SEQN"].is_unique)
check("No missing participant IDs", df["SEQN"].notna().all())
check("Expected column count", len(df.columns) == 658)
check(
    "Participant order preserved",
    source["SEQN"].equals(df["SEQN"])
)

mapping = {
    "serum_glucose_mg_dl": "BIOPRO__LBXSGL",
    "hba1c_percent": "GHB__LBXGH",
    "ferritin_ng_ml": "FERTIN__LBXFER",
    "serum_creatinine_mg_dl": "BIOPRO__LBXSCR",
    "alt_u_l": "BIOPRO__LBXSATSI",
    "sodium_mmol_l": "BIOPRO__LBXSNASI",
    "serum_calcium_mg_dl": "BIOPRO__LBXSCA",
    "triglycerides_mg_dl": "TRIGLY__LBXTR",
}

for feature, original in mapping.items():
    expected = pd.to_numeric(
        source[original], errors="coerce"
    ).to_numpy(dtype=float)

    actual = pd.to_numeric(
        df[feature], errors="coerce"
    ).to_numpy(dtype=float)

    check(
        f"Source match: {feature}",
        np.allclose(
            expected, actual,
            equal_nan=True,
            rtol=1e-10,
            atol=1e-10
        )
    )

for kind, prefix, upper in [
    ("systolic", "BPXSY", 300),
    ("diastolic", "BPXDI", 200),
]:
    cols = [
        f"BPX__{prefix}{i}"
        for i in range(1, 5)
    ]

    readings = source[cols].apply(
        pd.to_numeric, errors="coerce"
    )
    readings = readings.where(
        (readings > 0) & (readings <= upper)
    )

    expected_mean = readings.mean(axis=1).to_numpy()
    actual_mean = df[
        f"{kind}_bp_mean_mmhg"
    ].to_numpy()

    expected_count = readings.notna().sum(axis=1)
    actual_count = df[
        f"{kind}_bp_reading_count"
    ]

    check(
        f"{kind} BP mean",
        np.allclose(
            expected_mean,
            actual_mean,
            equal_nan=True
        )
    )

    check(
        f"{kind} BP reading count",
        expected_count.equals(actual_count)
    )

report = {
    "checks": checks,
    "passed": sum(checks.values()),
    "total": len(checks),
    "all_passed": all(checks.values()),
}

REPORTS.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 60)
print(
    f"VALIDATION: {report['passed']}/{report['total']}"
)
print("=" * 60)

if not report["all_passed"]:
    raise SystemExit("Validation failed")

print("All checks passed.")
print("Report:", OUTPUT)
