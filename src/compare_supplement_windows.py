
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data" / "processed" / "nutrition_30day_ml.csv"
REPORT_PATH = BASE / "reports" / "supplement_window_comparison.json"

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["LBXVIDMS"]).copy()

df["low_vitamin_d"] = (df["LBXVIDMS"] < 50).astype(int)

# Normalize tiny SAS floating-point representations of zero.
for column in [
    "supplement_count",
    "supplement_count_30day",
]:
    df.loc[df[column].abs() < 1e-20, column] = 0

# Preserve missingness as a separate predictor.
for amount, count, prefix in [
    (
        "supplement_vitamin_d_mcg",
        "supplement_count",
        "day1",
    ),
    (
        "vitamin_d_supplement_30day_mcg",
        "supplement_count_30day",
        "day30",
    ),
]:
    df[f"{prefix}_amount_missing"] = (
        df[amount].isna().astype(int)
    )

    # A reported zero supplement count supports zero intake.
    # For other missing amounts, leave them missing for
    # pipeline imputation instead of inventing zero values.
    zero_count = df[count].eq(0)
    df.loc[zero_count & df[amount].isna(), amount] = 0


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

base_categorical = [
    "sex_code",
    "examination_period",
]

feature_sets = {
    "Original + Season": base_numeric,
    "Day 1 + Season": base_numeric + [
        "supplement_vitamin_d_mcg",
        "supplement_count",
        "day1_amount_missing",
    ],
    "30 Day + Season": base_numeric + [
        "vitamin_d_supplement_30day_mcg",
        "supplement_count_30day",
        "day30_amount_missing",
    ],
    "Both Windows + Season": base_numeric + [
        "supplement_vitamin_d_mcg",
        "supplement_count",
        "day1_amount_missing",
        "vitamin_d_supplement_30day_mcg",
        "supplement_count_30day",
        "day30_amount_missing",
    ],
}

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


def build_pipeline(numeric_features, classifier):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True,
        )),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent",
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
        )),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, base_categorical),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1,
    ),
}

results = {}

print("\nSUPPLEMENT WINDOW COMPARISON")
print("Total records:", len(df))
print("Training records:", len(train_idx))

for feature_name, numeric_features in feature_sets.items():
    features = numeric_features + base_categorical
    X_train = df.iloc[train_idx][features]

    results[feature_name] = {}

    for model_name, classifier in models.items():
        pipeline = build_pipeline(
            numeric_features,
            classifier,
        )

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


REPORT_PATH.parent.mkdir(exist_ok=True, parents=True)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nCOMPARISON COMPLETE")
print("Saved:", REPORT_PATH)
