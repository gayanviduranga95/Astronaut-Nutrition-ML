"""Reusable leakage-safe infrastructure; callers must establish a legitimate target."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42

def build_research_classifier(numeric_features, categorical_features, target_provenance):
    if not target_provenance:
        raise ValueError("A documented legitimate target provenance is required")
    overlap = set(numeric_features) | set(categorical_features)
    if target_provenance.get("target_name") in overlap:
        raise ValueError("Target must not be included among model features")
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler())]), numeric_features),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
    ])
    return Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))])

def model_metadata(features, target_provenance):
    return {"random_seed": RANDOM_SEED, "features": list(features), "target_provenance": target_provenance,
            "preprocessing": "Pipeline fit on training split only; caller must split before fit.",
            "research_only": True, "external_or_astronaut_validation": False}
