"""Engineering validation utilities; plausibility checks are not clinical ranges."""
import math
from collections import Counter


ENGINEERING_LIMITS = {
    "bmi": (1, 100), "systolic_bp_mean_mmhg": (0, 300),
    "diastolic_bp_mean_mmhg": (0, 200), "hba1c_percent": (0, 25),
    "sodium_mmol_l": (0, 250), "potassium_mmol_l": (0, 20),
    "chloride_mmol_l": (0, 200),
}


def finite_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def validate_measurements(measurements, limits=None):
    """Return cleaned values and explicit engineering validation issues."""
    limits = limits or ENGINEERING_LIMITS
    cleaned, issues = {}, []
    for name, raw in measurements.items():
        value = finite_number(raw)
        if value is None:
            issues.append({"measurement": name, "issue": "missing_or_non_finite"})
            continue
        if name in limits and not limits[name][0] <= value <= limits[name][1]:
            issues.append({"measurement": name, "issue": "outside_engineering_plausibility_range"})
            continue
        cleaned[name] = value
    return cleaned, issues


def duplicate_values(values):
    counts = Counter(values)
    return {str(value): count for value, count in counts.items() if count > 1}


def valid_denominator(value):
    value = finite_number(value)
    return value is not None and value > 0
