def generate_insights(assessment, longitudinal=None):
    insights = []
    for domain, item in assessment["domains"].items():
        for finding in item["findings"]:
            insights.append({"domain": domain, "measurement": finding["measurement"], "observed_value": finding["value"],
                             "calculation_method": "published_research_reference_comparison", "source": finding["reference"],
                             "limitations": "Research comparison only; not a diagnosis or astronaut-specific interpretation."})
    for trend in (longitudinal or {}).get("trends", []):
        for domain in trend["domains"]:
            insights.append({"domain": domain, "measurement": trend["measurement"], "observed_value": trend["latest_value"],
                             "absolute_change": trend["absolute_change"], "calculation_method": trend["calculation_method"],
                             "source": "timestamped observation input", "limitations": "Descriptive observed change only."})
    return {"insights": insights, "insight_count": len(insights)}
