
from typing import Any


# Demonstration criteria for adult research use.
# These are not astronaut-specific clinical ranges.
# Clinical use requires source verification, age/sex
# applicability, and appropriate measurement context.

REFERENCE_RULES = {
    "hba1c_percent": {
        "lower": 5.7,
        "upper": 6.5,
        "labels": [
            "below_prediabetes_threshold",
            "prediabetes_range",
            "diabetes_range",
        ],
        "source": "ADA adult HbA1c diagnostic criteria",
    },
    "total_cholesterol_mg_dl": {
        "lower": 200,
        "upper": 240,
        "labels": [
            "desirable_range",
            "borderline_high_range",
            "high_range",
        ],
        "source": "NCEP adult total cholesterol categories",
    },
    "triglycerides_mg_dl": {
        "lower": 150,
        "upper": 200,
        "labels": [
            "normal_range",
            "borderline_high_range",
            "high_or_above_range",
        ],
        "source": "NCEP adult triglyceride categories",
    },
}


def interpret_measurement(
    name: str,
    value: Any,
):
    if name not in REFERENCE_RULES:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    rule = REFERENCE_RULES[name]

    if number < rule["lower"]:
        label = rule["labels"][0]
    elif number < rule["upper"]:
        label = rule["labels"][1]
    else:
        label = rule["labels"][2]

    return {
        "measurement": name,
        "value": number,
        "interpretation": label,
        "reference": rule["source"],
        "diagnosis": False,
        "research_only": True,
    }


def interpret_nutrition(measurements: dict):
    findings = []

    for name, value in measurements.items():
        finding = interpret_measurement(name, value)

        if finding is not None:
            findings.append(finding)

    return {
        "interpretation_version": "0.1-research",
        "findings": findings,
        "finding_count": len(findings),
        "disclaimer": (
            "Research-only reference comparisons. "
            "Not a diagnosis or validated astronaut "
            "health assessment."
        ),
    }


if __name__ == "__main__":
    sample = {
        "hba1c_percent": 5.9,
        "total_cholesterol_mg_dl": 215,
        "triglycerides_mg_dl": 175,
    }

    import json

    print(
        json.dumps(
            interpret_nutrition(sample),
            indent=2,
        )
    )
