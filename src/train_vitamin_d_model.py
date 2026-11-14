
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ==========================================
# 1. PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "nutrition_with_labs.csv"
)

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

print("Original records:", len(df))

# Use only participants with measured vitamin D.
df = df.dropna(subset=["LBXVIDMS"]).copy()

# Exploratory binary target.
# 1 = measured serum vitamin D below 50 nmol/L.
df["low_vitamin_d"] = (
    df["LBXVIDMS"] < 50
).astype(int)

print("Records with vitamin D:", len(df))

print("\nTarget distribution:")
print(df["low_vitamin_d"].value_counts())


# ==========================================
# 3. SELECT FEATURES
# ==========================================

# Only use information that could reasonably
# be available without laboratory results.
#
# Do not include LBXVIDMS or other lab values.
# Do not include SEQN (participant identifier).

numeric_features = [
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

categorical_features = [
    "sex_code",
]

features = numeric_features + categorical_features

X = df[features].copy()
y = df["low_vitamin_d"]


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 5. PREPROCESSING
# ==========================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ==========================================
# 6. MODELS
# ==========================================

models = {
    "Dummy Baseline": DummyClassifier(
        strategy="prior"
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
}


# ==========================================
# 7. TRAIN AND EVALUATE
# ==========================================

results = {}
trained_models = {}

for name, model in models.items():

    print("\n" + "=" * 55)
    print("MODEL:", name)
    print("=" * 55)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_test, predictions
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test, probabilities
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_test, probabilities
            )
        )
    }

    results[name] = metrics
    trained_models[name] = pipeline

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test, predictions
        )
    )


# ==========================================
# 8. SAVE EXPERIMENT RESULTS
# ==========================================

report_path = (
    REPORT_DIR / "vitamin_d_model_results.json"
)

with open(report_path, "w") as f:
    json.dump(results, f, indent=2)

# Save the candidate with the highest test ROC-AUC
# for research/demo use only.
# A separate validation set or cross-validation
# is needed for unbiased model selection.

best_name = max(
    results,
    key=lambda name: results[name]["roc_auc"]
)

best_model = trained_models[best_name]

model_path = (
    MODEL_DIR / "vitamin_d_research_model.joblib"
)

joblib.dump(
    {
        "model": best_model,
        "features": features,
        "target": "low_vitamin_d",
        "threshold_nmol_l": 50,
        "model_name": best_name,
        "research_only": True
    },
    model_path
)

print("\n" + "=" * 55)
print("TRAINING COMPLETED")
print("=" * 55)

print("Highest test ROC-AUC model:", best_name)
print("Model saved:", model_path)
print("Results saved:", report_path)

print(
    "\nWARNING: This is a research-only model. "
    "Do not use it for clinical or astronaut "
    "health decisions."
)
