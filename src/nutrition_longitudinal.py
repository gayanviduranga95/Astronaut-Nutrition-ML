from datetime import datetime
from src.nutrition_engine_v2 import DOMAIN_FEATURES
from src.nutrition_validation import finite_number


def _domains_for(measurement):
    return [domain for domain, fields in DOMAIN_FEATURES.items() if measurement in fields]


def summarize_observations(subject_id, observations):
    """Summarize repeated records; never treats independent NHANES rows as a trajectory."""
    if not observations:
        raise ValueError("At least one observation is required")
    parsed, timestamps = [], set()
    for observation in observations:
        if observation.get("subject_id") != subject_id:
            raise ValueError("Every observation must use the requested subject_id")
        timestamp = datetime.fromisoformat(observation["timestamp"].replace("Z", "+00:00"))
        if timestamp in timestamps:
            raise ValueError("Duplicate observation timestamps are not allowed")
        timestamps.add(timestamp)
        parsed.append((timestamp, observation))
    parsed.sort(key=lambda item: item[0])
    measurements = sorted({key for _, row in parsed for key in row.get("measurements", {})})
    trends = []
    for name in measurements:
        values = [(time, finite_number(row.get("measurements", {}).get(name))) for time, row in parsed]
        values = [(time, value) for time, value in values if value is not None]
        if not values:
            continue
        first_time, first = values[0]; latest_time, latest = values[-1]
        change = latest - first
        trends.append({"measurement": name, "domains": _domains_for(name), "first_value": first,
                       "latest_value": latest, "absolute_change": change,
                       "percentage_change": None if first == 0 else round(change / abs(first) * 100, 2),
                       "measurement_count": len(values), "interval_days": round((latest_time - first_time).total_seconds() / 86400, 4),
                       "calculation_method": "latest_valid_value_minus_first_valid_value"})
    return {"subject_id": subject_id, "observation_count": len(parsed), "trends": trends,
            "limitations": ["Observed changes are not clinical improvement or deterioration.",
                            "Independent NHANES participants must not be used as repeated observations."]}
