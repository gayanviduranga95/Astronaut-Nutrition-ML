
from typing import Dict, Any


NUTRIENTS = {
    "energy_kcal": "Energy",
    "protein_g": "Protein",
    "carbohydrate_g": "Carbohydrates",
    "fat_g": "Fat",
    "calcium_mg": "Calcium",
    "iron_mg": "Iron",
    "vitamin_d_mcg": "Vitamin D",
    "fiber_g": "Fiber",
    "vitamin_b12_mcg": "Vitamin B12",
    "vitamin_c_mg": "Vitamin C",
    "magnesium_mg": "Magnesium",
    "zinc_mg": "Zinc",
    "potassium_mg": "Potassium",
}


def assess_multi_nutrient(
    intakes: Dict[str, float],
    targets: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compare nutrient intakes against user-supplied
    reference targets.

    Targets must come from an appropriate mission
    nutrition plan. This is not a clinical diagnosis.
    """

    results = []
    missing = []

    for key, name in NUTRIENTS.items():
        if key not in intakes or key not in targets:
            missing.append(key)
            continue

        intake = float(intakes[key])
        target = float(targets[key])

        if not (
            0 <= intake < float("inf")
            and 0 < target < float("inf")
        ):
            raise ValueError(
                f"Invalid intake or target for {key}"
            )

        ratio = intake / target
        percentage = ratio * 100
        shortfall = max(0, target - intake)

        if percentage < 70:
            status = "large_shortfall"
        elif percentage < 90:
            status = "moderate_shortfall"
        elif percentage < 100:
            status = "small_shortfall"
        else:
            status = "target_met_or_exceeded"

        results.append({
            "nutrient": key,
            "name": name,
            "intake": round(intake, 2),
            "target": round(target, 2),
            "percentage_of_target": round(
                percentage, 2
            ),
            "shortfall": round(shortfall, 2),
            "status": status,
        })

    results.sort(
        key=lambda item: item["percentage_of_target"]
    )

    assessed = len(results)
    below_target = sum(
        item["shortfall"] > 0
        for item in results
    )

    return {
        "assessed_nutrients": assessed,
        "below_target_count": below_target,
        "results": results,
        "missing_nutrients": missing,
        "disclaimer": (
            "Intake comparison only. Targets must be "
            "provided by an appropriate nutrition plan. "
            "Shortfalls are not confirmed deficiencies. "
            "Meeting a minimum target does not establish "
            "that an intake is safe or optimal."
        ),
    }


if __name__ == "__main__":
    example_intakes = {
        "energy_kcal": 2100,
        "protein_g": 80,
        "carbohydrate_g": 250,
        "fat_g": 70,
        "calcium_mg": 900,
        "iron_mg": 12,
        "vitamin_d_mcg": 5,
        "fiber_g": 20,
        "vitamin_b12_mcg": 3,
        "vitamin_c_mg": 75,
        "magnesium_mg": 320,
        "zinc_mg": 10,
        "potassium_mg": 2600,
    }

    example_targets = {
        "energy_kcal": 2500,
        "protein_g": 100,
        "carbohydrate_g": 300,
        "fat_g": 80,
        "calcium_mg": 1000,
        "iron_mg": 15,
        "vitamin_d_mcg": 15,
        "fiber_g": 30,
        "vitamin_b12_mcg": 2.4,
        "vitamin_c_mg": 90,
        "magnesium_mg": 400,
        "zinc_mg": 11,
        "potassium_mg": 3400,
    }

    import json

    result = assess_multi_nutrient(
        example_intakes,
        example_targets,
    )

    print(json.dumps(result, indent=2))
