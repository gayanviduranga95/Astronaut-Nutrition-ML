
import json
import math
from typing import Any


# Existing adult research reference comparisons.
# Not astronaut-specific diagnostic criteria.
REFERENCE_RULES = {
    "hba1c_percent": {
        "lower": 5.7,
        "upper": 6.5,
        "labels": [
            "below_prediabetes_threshold",
            "prediabetes_range",
            "diabetes_range",
        ],
        "source": "ADA adult HbA1c criteria",
        "domain": "glucose_regulation",
    },
    "total_cholesterol_mg_dl": {
        "lower": 200,
        "upper": 240,
        "labels": [
            "desirable_range",
            "borderline_high_range",
            "high_range",
        ],
        "source": "NCEP adult cholesterol categories",
        "domain": "cardiovascular_metabolic",
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
        "domain": "cardiovascular_metabolic",
    },
}


# These definitions intentionally mirror the engine's nine monitoring domains.
# Shared measurements are included in every domain to which they contribute.
DOMAIN_MEASUREMENTS = {
    "nutritional_status": (
        "avg_protein_g", "avg_iron_mg", "bmi", "albumin_g_dl",
    ),
    "anemia_iron": (
        "hemoglobin_g_dl", "mcv_fl", "ferritin_ng_ml", "serum_iron_ug_dl",
    ),
    "glucose_regulation": ("serum_glucose_mg_dl", "hba1c_percent"),
    "kidney_health": (
        "serum_creatinine_mg_dl", "bun_mg_dl", "urine_albumin_ug_ml",
        "urine_creatinine_mg_dl",
    ),
    "kidney_stone": ("serum_calcium_mg_dl", "serum_phosphorus_mg_dl"),
    "liver_health": (
        "alt_u_l", "ast_u_l", "albumin_g_dl", "total_bilirubin_mg_dl",
    ),
    "cardiovascular_metabolic": (
        "total_cholesterol_mg_dl", "hdl_cholesterol_mg_dl",
        "triglycerides_mg_dl", "systolic_bp_mean_mmhg",
        "diastolic_bp_mean_mmhg",
    ),
    "hydration_electrolytes": (
        "sodium_mmol_l", "potassium_mmol_l", "chloride_mmol_l",
    ),
    "bone_nutrition": (
        "serum_vitamin_d_nmol_l", "serum_calcium_mg_dl",
        "serum_phosphorus_mg_dl",
    ),
}

MEASUREMENT_DOMAINS = {
    measurement: tuple(
        domain for domain, measurements in DOMAIN_MEASUREMENTS.items()
        if measurement in measurements
    )
    for measurements in DOMAIN_MEASUREMENTS.values()
    for measurement in measurements
}


def number_or_none(value: Any):
    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    return number if math.isfinite(number) else None


def reference_finding(name, value):
    rule = REFERENCE_RULES.get(name)

    if rule is None:
        return None

    if value < rule["lower"]:
        label = rule["labels"][0]
    elif value < rule["upper"]:
        label = rule["labels"][1]
    else:
        label = rule["labels"][2]

    return {
        "measurement": name,
        "value": value,
        "interpretation": label,
        "reference": rule["source"],
        "domain": rule["domain"],
        "finding_type": "reference_comparison",
        "diagnosis": False,
        "research_only": True,
    }


def calculate_urine_acr(measurements):
    """
    Urine albumin-to-creatinine ratio.

    Albumin: micrograms/mL
    Creatinine: mg/dL

    ACR (mg/g) = albumin * 100 / creatinine.

    The conversion follows the supplied NHANES-style units: ug/mL albumin is
    numerically mg/L, while mg/dL creatinine is 0.01 g/L. Their ratio is thus
    albumin * 100 / creatinine in mg/g.

    Only calculated when both inputs are positive.
    """
    albumin = number_or_none(
        measurements.get("urine_albumin_ug_ml")
    )
    creatinine = number_or_none(
        measurements.get("urine_creatinine_mg_dl")
    )

    if (
        albumin is None
        or creatinine is None
        or albumin < 0
        or creatinine <= 0
    ):
        return None

    acr = albumin * 100 / creatinine
    return acr if math.isfinite(acr) else None


def interpret_nutrition(measurements: dict):
    if not isinstance(measurements, dict):
        raise TypeError("measurements must be a dictionary")

    findings = []
    summaries = []

    for name, raw_value in measurements.items():
        value = number_or_none(raw_value)

        if value is None:
            continue

        finding = reference_finding(name, value)

        if finding is not None:
            findings.append(finding)

        for domain in MEASUREMENT_DOMAINS.get(name, ()):
            summaries.append({
                "measurement": name,
                "value": value,
                "domain": domain,
                "interpretation": "measurement_recorded",
                "finding_type": "descriptive",
                "diagnosis": False,
            })

    acr = calculate_urine_acr(measurements)

    derived = {}

    if acr is not None:
        derived["urine_acr_mg_g"] = round(acr, 2)

        summaries.append({
            "measurement": "urine_acr_mg_g",
            "value": round(acr, 2),
            "domain": "kidney_health",
            "interpretation": "calculated_not_classified",
            "finding_type": "derived_measurement",
            "diagnosis": False,
        })

    return {
        "interpretation_version": "0.2-research",
        "findings": findings,
        "finding_count": len(findings),
        "measurement_summaries": summaries,
        "derived_measurements": derived,
        "disclaimer": (
            "Research-only reference comparisons and "
            "descriptive measurements. No diagnosis or "
            "validated astronaut health prediction."
        ),
    }


if __name__ == "__main__":
    sample = {
        "hba1c_percent": 5.9,
        "total_cholesterol_mg_dl": 215,
        "triglycerides_mg_dl": 175,
        "hemoglobin_g_dl": 13.5,
        "ferritin_ng_ml": 45,
        "serum_creatinine_mg_dl": 0.9,
        "urine_albumin_ug_ml": 15,
        "urine_creatinine_mg_dl": 100,
        "sodium_mmol_l": 140,
        "serum_vitamin_d_nmol_l": 60,
    }

    print(json.dumps(
        interpret_nutrition(sample),
        indent=2
    ))
