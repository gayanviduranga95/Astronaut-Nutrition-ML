# Astronaut Nutrition ML

Research software for descriptive nutrition monitoring in Medora AI. It uses NHANES Earth-population data; it is not trained, validated, or clinically ready for astronaut health assessment.

## Architecture

- `nutrition_engine_v2`: nine-domain laboratory and measurement availability assessment.
- `nutrition_interpretation`: descriptive summaries, three sourced research comparisons, and urine ACR calculation.
- `nutrition_quality`: read-only V3 data audit and ML feasibility reporting.
- `nutrition_dietary`, `nutrition_longitudinal`, and `nutrition_insights`: context-aware intake comparison, timestamped monitoring, and traceable descriptive insights.

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`. Existing routes remain available, including `/nutrition-v2/health`, `/nutrition-v2/assess`, and `/assess-nutrition`.

New versioned research routes are `/nutrition-v3/data-quality`, `/nutrition-v3/ml-feasibility`, `/nutrition-v3/dietary-assess`, `/nutrition-v3/longitudinal-assess`, `/nutrition-v3/insights`, and `/nutrition-v3/research-report`.

Example:

```json
{"measurements":{"hba1c_percent":5.9,"urine_albumin_ug_ml":15,"urine_creatinine_mg_dl":100}}
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Limitations

The service reports descriptive observations, limited adult research reference comparisons, and input-derived calculations. It provides no diagnosis, disease-risk score, causal inference, clinical advice, fabricated outcomes, or astronaut-specific prediction. One-day dietary recalls do not establish habitual intake. The ML feasibility report deliberately does not train a new domain predictor without a verified target and leakage-safe study design.
