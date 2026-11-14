
import json
import math
from typing import Dict


SUPPORTED_NUTRIENTS = {
    "energy_kcal",
    "protein_g",
    "carbohydrate_g",
    "fat_g",
    "calcium_mg",
    "iron_mg",
    "vitamin_d_mcg",
    "fiber_g",
    "vitamin_b12_mcg",
    "vitamin_c_mg",
    "magnesium_mg",
    "zinc_mg",
    "potassium_mg",
}


def assess_nutrition(
    intakes: Dict[str, float],
    requirements: Dict[str, float]
) -> dict:

    if not intakes:
        raise ValueError("At least one nutrient is required.")

    if set(intakes) != set(requirements):
        raise ValueError(
            "Intakes and requirements must have matching nutrients."
        )

    unknown = set(intakes) - SUPPORTED_NUTRIENTS

    if unknown:
        raise ValueError(
            f"Unsupported nutrients: {sorted(unknown)}"
        )

    results = {}
    below_target = []

    for nutrient, actual in intakes.items():

        required = requirements[nutrient]

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in (actual, required)
        ):
            raise ValueError(
                f"Invalid numeric value for {nutrient}"
            )

        if actual < 0 or required <= 0:
            raise ValueError(
                f"Invalid intake or requirement for {nutrient}"
            )

        adequacy_pct = actual / required * 100

        results[nutrient] = {
            "actual": round(actual, 2),
            "required": round(required, 2),
            "adequacy_percent": round(adequacy_pct, 1),
            "below_target": actual < required
        }

        if actual < required:
            below_target.append(nutrient)

    return {
        "assessment_type": "nutrient_adequacy_screening",
        "nutrients": results,
        "below_target_nutrients": below_target,
        "below_target_count": len(below_target),
        "interpretation": (
            "Some reported intakes are below configured targets."
            if below_target
            else "All reported intakes meet configured targets."
        ),
        "disclaimer": (
            "This is an intake-versus-target comparison, "
            "not a clinical malnutrition diagnosis or "
            "validated astronaut risk prediction."
        )
    }


if __name__ == "__main__":

    # Example inputs for testing only.
    # These are NOT validated astronaut requirements.
    intakes = {
        "energy_kcal": 1800,
        "protein_g": 65,
        "carbohydrate_g": 220,
        "fat_g": 70,
        "calcium_mg": 800,
        "iron_mg": 12,
        "vitamin_d_mcg": 5,
        "fiber_g": 15,
        "vitamin_b12_mcg": 3,
        "vitamin_c_mg": 60,
        "magnesium_mg": 260,
        "zinc_mg": 8,
        "potassium_mg": 2300
    }

    requirements = {
        "energy_kcal": 2400,
        "protein_g": 90,
        "carbohydrate_g": 250,
        "fat_g": 80,
        "calcium_mg": 1000,
        "iron_mg": 15,
        "vitamin_d_mcg": 15,
        "fiber_g": 30,
        "vitamin_b12_mcg": 2.4,
        "vitamin_c_mg": 90,
        "magnesium_mg": 400,
        "zinc_mg": 11,
        "potassium_mg": 3400
    }

    result = assess_nutrition(intakes, requirements)
    print(json.dumps(result, indent=2))
