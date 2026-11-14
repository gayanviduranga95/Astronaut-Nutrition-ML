
import json
import math
from typing import Any

import pandas as pd

try:
    from src.nutrition_interpretation import interpret_nutrition
except ModuleNotFoundError:
    from nutrition_interpretation import interpret_nutrition


# ============================================================
# 1. NINE NUTRITION HEALTH DOMAINS
# ============================================================

DOMAIN_FEATURES = {
    "nutritional_status": [
        "avg_protein_g",
        "avg_iron_mg",
        "bmi",
        "albumin_g_dl",
    ],
    "anemia_iron": [
        "hemoglobin_g_dl",
        "mcv_fl",
        "ferritin_ng_ml",
        "serum_iron_ug_dl",
    ],
    "glucose_regulation": [
        "serum_glucose_mg_dl",
        "hba1c_percent",
    ],
    "kidney_health": [
        "serum_creatinine_mg_dl",
        "bun_mg_dl",
        "urine_albumin_ug_ml",
        "urine_creatinine_mg_dl",
    ],
    "kidney_stone": [
        "serum_calcium_mg_dl",
        "serum_phosphorus_mg_dl",
    ],
    "liver_health": [
        "alt_u_l",
        "ast_u_l",
        "albumin_g_dl",
        "total_bilirubin_mg_dl",
    ],
    "cardiovascular_metabolic": [
        "total_cholesterol_mg_dl",
        "hdl_cholesterol_mg_dl",
        "triglycerides_mg_dl",
        "systolic_bp_mean_mmhg",
        "diastolic_bp_mean_mmhg",
    ],
    "hydration_electrolytes": [
        "sodium_mmol_l",
        "potassium_mmol_l",
        "chloride_mmol_l",
    ],
    "bone_nutrition": [
        "serum_vitamin_d_nmol_l",
        "serum_calcium_mg_dl",
        "serum_phosphorus_mg_dl",
    ],
}


# ============================================================
# 2. FINDING-TO-DOMAIN MAPPING
# ============================================================

FINDING_DOMAIN_MAP = {
    "hba1c_percent": "glucose_regulation",
    "total_cholesterol_mg_dl": "cardiovascular_metabolic",
    "triglycerides_mg_dl": "cardiovascular_metabolic",
}


# ============================================================
# 3. DOMAIN LIMITATIONS
# ============================================================

LIMITED_DOMAINS = {
    "kidney_stone": (
        "Serum calcium and phosphorus alone cannot "
        "establish kidney-stone risk. Dedicated urine "
        "chemistry and clinical context are needed."
    ),
    "hydration_electrolytes": (
        "Serum electrolytes alone cannot establish "
        "hydration status or fluid balance."
    ),
    "bone_nutrition": (
        "Nutritional biomarkers alone cannot establish "
        "bone density or fracture risk."
    ),
}


DEFAULT_LIMITATION = (
    "Measurement completeness and research reference "
    "comparisons only. No clinical diagnosis or "
    "validated astronaut risk prediction is provided."
)


# ============================================================
# 4. NUMERIC VALUE CLEANING
# ============================================================

def clean_number(value: Any):
    """
    Convert numeric measurements into JSON-safe floats.

    Missing, invalid, and non-finite values become None.
    Boolean values are not accepted as measurements.
    """

    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(number):
        return None

    return number


# ============================================================
# 5. ASSESS INDIVIDUAL HEALTH DOMAIN
# ============================================================

def assess_domain(record, domain, features):
    measurements = {
        name: clean_number(record.get(name))
        for name in features
    }

    available = {
        name: value
        for name, value in measurements.items()
        if value is not None
    }

    missing = [
        name
        for name, value in measurements.items()
        if value is None
    ]

    if not available:
        status = "insufficient_data"

    elif domain in LIMITED_DOMAINS:
        status = "limited_assessment"

    elif missing:
        status = "insufficient_data"

    else:
        status = "data_available"

    return {
        "status": status,
        "available_count": len(available),
        "required_count": len(features),
        "measurements": available,
        "missing_features": missing,
        "clinical_risk_assessed": False,
        "limitation": LIMITED_DOMAINS.get(
            domain,
            DEFAULT_LIMITATION,
        ),
        "findings": [],
        "finding_count": 0,
        "measurement_summaries": [],
        "derived_measurements": {},
        "interpretation_summary": None,
    }


def domain_interpretation_summary(assessment):
    """Describe measurement coverage without producing clinical labels."""
    available_count = assessment["available_count"]
    required_count = assessment["required_count"]
    reference_count = assessment["finding_count"]

    if available_count == 0:
        return (
            "No supported measurements were provided for this domain; "
            "no interpretation is available."
        )

    summary = (
        f"{available_count} of {required_count} supported measurements are "
        "available for descriptive monitoring"
    )
    if reference_count:
        summary += f"; {reference_count} research reference comparison(s) are included"
    summary += "."

    if assessment["missing_features"]:
        summary += " Incomplete measurements limit interpretation."
    if assessment["status"] == "limited_assessment":
        summary += " This domain has additional measurement limitations."
    return summary


# ============================================================
# 6. INTEGRATED NUTRITION ASSESSMENT
# ============================================================

def assess_nutrition(record: dict):
    """
    Assess one nutrition/laboratory record.

    Returns:
    - Nine health domains
    - Measurement availability
    - Missing measurements
    - Research reference findings
    - Findings grouped by domain
    """

    if not isinstance(record, dict):
        raise TypeError(
            "Nutrition record must be a dictionary"
        )

    results = {}

    # Assess all nine domains.
    for domain, features in DOMAIN_FEATURES.items():
        results[domain] = assess_domain(
            record,
            domain,
            features,
        )

    # Interpret supported measurements.
    interpretation = interpret_nutrition(record)

    # Attach each finding to its health domain. The interpretation-provided
    # domain is preferred; the map remains as a backwards-compatible fallback.
    for finding in interpretation["findings"]:
        measurement = finding["measurement"]

        domain = finding.get("domain") or FINDING_DOMAIN_MAP.get(measurement)

        if domain in results:
            results[domain]["findings"].append(
                finding.copy()
            )

    # Keep descriptive and derived measurements separate from research findings.
    # This avoids treating routine recorded values as clinical findings.
    for summary in interpretation["measurement_summaries"]:
        domain = summary.get("domain")
        if domain in results:
            results[domain]["measurement_summaries"].append(summary.copy())
            if summary.get("finding_type") == "derived_measurement":
                results[domain]["derived_measurements"][summary["measurement"]] = (
                    summary["value"]
                )

    # Calculate finding counts.
    for domain in results:
        results[domain]["finding_count"] = len(
            results[domain]["findings"]
        )
        results[domain]["interpretation_summary"] = domain_interpretation_summary(
            results[domain]
        )

    return {
        "system": "Astronaut Nutrition ML",
        "engine_version": "2.2.0-research",
        "assessment_type": (
            "integrated_nutrition_assessment"
        ),
        "domains": results,
        "interpretation": interpretation,
        "disclaimer": (
            "Research and engineering prototype. "
            "Reference comparisons are not medical "
            "diagnoses or validated astronaut health "
            "risk predictions."
        ),
    }


# ============================================================
# 7. BATCH DATASET ASSESSMENT
# ============================================================

def assess_dataframe(df: pd.DataFrame):
    """
    Assess multiple participant records.

    Returns one assessment per dataset row.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame"
        )

    return [
        assess_nutrition(record)
        for record in df.to_dict(
            orient="records"
        )
    ]


# ============================================================
# 8. LOCAL TEST
# ============================================================

if __name__ == "__main__":

    sample = {
        "avg_protein_g": 80,
        "avg_iron_mg": 14,
        "bmi": 23.5,
        "albumin_g_dl": 4.2,

        "serum_glucose_mg_dl": 95,
        "hba1c_percent": 5.9,

        "total_cholesterol_mg_dl": 215,
        "triglycerides_mg_dl": 175,

        "sodium_mmol_l": 140,
        "potassium_mmol_l": 4.2,
        "chloride_mmol_l": 103,
    }

    result = assess_nutrition(sample)

    print("=" * 65)
    print("ASTRONAUT NUTRITION ML V2.2")
    print("=" * 65)

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print("\n" + "=" * 65)
    print("DOMAIN FINDING SUMMARY")
    print("=" * 65)

    for domain, assessment in result["domains"].items():
        print(
            f"{domain}: "
            f"{assessment['finding_count']} findings"
        )

    print(
        "\nTotal research findings:",
        result["interpretation"]["finding_count"],
    )
