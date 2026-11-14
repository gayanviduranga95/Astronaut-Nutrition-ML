"""Descriptive dietary intake comparison with caller-supplied references."""
from src.nutrition_validation import finite_number, valid_denominator


DIETARY_FIELDS = (
    "energy_kcal", "protein_g", "carbohydrate_g", "fat_g", "fiber_g",
    "iron_mg", "calcium_mg", "sodium_mg", "potassium_mg", "vitamin_d_mcg",
    "vitamin_b12_mcg", "vitamin_c_mg", "magnesium_mg", "zinc_mg",
)


def assess_dietary_intake(intakes, references=None, context=None):
    """One-day observed intake is not treated as habitual intake or an astronaut requirement."""
    references, context = references or {}, context or {}
    results, missing_context = [], []
    for name, raw in intakes.items():
        if name not in DIETARY_FIELDS:
            continue
        intake = finite_number(raw)
        if intake is None or intake < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
        item = {"measurement": name, "observed_intake": intake, "reference_intake": None,
                "percentage_of_reference": None, "comparison_status": "insufficient_reference_context"}
        reference = finite_number(references.get(name))
        if reference is not None and valid_denominator(reference) and context:
            item.update(reference_intake=reference, percentage_of_reference=round(intake / reference * 100, 2),
                        comparison_status="user_supplied_reference_compared")
        elif reference is not None:
            missing_context.append(name)
        results.append(item)
    return {"assessment_type": "descriptive_dietary_intake", "results": results,
            "context_supplied": bool(context), "insufficient_context_for": missing_context,
            "limitations": ["Reported intake may represent one NHANES recall day, not habitual intake.",
                            "References are caller-supplied and are not validated astronaut requirements."]}
