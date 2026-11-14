
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data" / "processed" / "nutrition_enhanced_ml.csv"
REPORT_PATH = BASE / "reports" / "feature_contributions.json"

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["LBXVIDMS"]).copy()

df["low_vitamin_d"] = (df["LBXVIDMS"] < 50).astype(int)

df["supplement_vitamin_d_missing"] = (
    df["supplement_vitamin_d_mcg"].isna().astype(int)
)

base_numeric = [
    "age_years",
    "bmi",
    "avg_energy_kcal",
    "avg_protein_g",
    "avg_fat_g",
    "avg_calcium_mg",
    "avg_vitamin_d_mcg",
    "avg_magnesium_mg",
    "avg_zinc_mg",
]

base_categorical = ["sex_code"]

supplement_numeric = [
    "supplement_vitamin_d_mcg",
    "supplement_count",
    "supplement_vitamin_d_missing",
]

feature_sets = {
    "Original": {
        "numeric": base_numeric,
        "categorical": base_categorical,
    },
    "Original + Season": {
        "numeric": base_numeric,
        "categorical": base_categorical + ["examination_period"],
    },
    "Original + Supplements": {
        "numeric": base_numeric + supplement_numeric,
        "categorical": base_categorical,
    },
    "All Enhanced": {
        "numeric": base_numeric + supplement_numeric,
        "categorical": base_categorical + ["examination_period"],
    },
}

# Preserve the same development split as previous experiments.
train_idx, _ = train_test_split(
    np.arange(len(df)),
    test_size=0.20,
    random_state=42,
    stratify=df["low_vitamin_d"],
)

y_train = df["low_vitamin_d"].iloc[train_idx]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


def build_pipeline(config, model):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True,
        )),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_transformer, config["numeric"]),
        ("categorical", categorical_transformer, config["categorical"]),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", model),
    ])


def candidate_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }


results = {}

print("\nFEATURE CONTRIBUTION EXPERIMENT")
print("Training participants:", len(train_idx))
print("Cross-validation folds: 5")

for feature_name, config in feature_sets.items():
    columns = config["numeric"] + config["categorical"]
    X_train = df.iloc[train_idx][columns]

    results[feature_name] = {}

    for model_name, model in candidate_models().items():
        pipeline = build_pipeline(config, model)

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "roc_auc": "roc_auc",
                "average_precision": "average_precision",
                "balanced_accuracy": "balanced_accuracy",
            },
            n_jobs=-1,
        )

        metrics = {
            "roc_auc_mean": round(
                float(scores["test_roc_auc"].mean()), 4
            ),
            "roc_auc_std": round(
                float(scores["test_roc_auc"].std()), 4
            ),
            "average_precision_mean": round(
                float(scores["test_average_precision"].mean()), 4
            ),
            "balanced_accuracy_mean": round(
                float(scores["test_balanced_accuracy"].mean()), 4
            ),
        }

        results[feature_name][model_name] = metrics

        print(f"\n{feature_name} | {model_name}")
        print(json.dumps(metrics, indent=2))


REPORT_PATH.parent.mkdir(exist_ok=True)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nEXPERIMENT COMPLETE")
print("Saved:", REPORT_PATH)
