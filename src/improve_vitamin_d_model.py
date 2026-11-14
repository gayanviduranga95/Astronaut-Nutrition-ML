
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
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed" / "nutrition_with_labs.csv"
MODELS = BASE / "models"
REPORTS = BASE / "reports"

MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df = df.dropna(subset=["LBXVIDMS"]).copy()

df["low_vitamin_d"] = (df["LBXVIDMS"] < 50).astype(int)

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

categorical_features = ["sex_code"]
features = numeric_features + categorical_features

X = df[features]
y = df["low_vitamin_d"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

def build_pipeline(classifier):
    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


candidates = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

cv_results = {}
best_name = None
best_auc = -1

for name, classifier in candidates.items():
    print(f"\nEvaluating {name}...")

    pipeline = build_pipeline(classifier)

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

    mean_auc = float(scores["test_roc_auc"].mean())

    cv_results[name] = {
        "roc_auc_mean": mean_auc,
        "roc_auc_std": float(scores["test_roc_auc"].std()),
        "average_precision_mean": float(
            scores["test_average_precision"].mean()
        ),
        "balanced_accuracy_mean": float(
            scores["test_balanced_accuracy"].mean()
        ),
    }

    print(json.dumps(cv_results[name], indent=2))

    if mean_auc > best_auc:
        best_auc = mean_auc
        best_name = name


print("\nSelected model:", best_name)

selected_pipeline = build_pipeline(candidates[best_name])

# Out-of-fold predictions from training data only.
oof_probabilities = cross_val_predict(
    selected_pipeline,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]

thresholds = np.arange(0.10, 0.91, 0.05)

threshold_results = []

for threshold in thresholds:
    predicted = (oof_probabilities >= threshold).astype(int)

    threshold_results.append({
        "threshold": round(float(threshold), 2),
        "recall": float(recall_score(y_train, predicted)),
        "precision": float(
            precision_score(y_train, predicted, zero_division=0)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_train, predicted)
        ),
    })

# Exploratory screening objective:
# Among thresholds with recall >= 0.80, select the one
# with the highest balanced accuracy.
eligible = [
    item for item in threshold_results
    if item["recall"] >= 0.80
]

if eligible:
    chosen = max(
        eligible,
        key=lambda item: item["balanced_accuracy"]
    )
else:
    chosen = max(
        threshold_results,
        key=lambda item: item["recall"]
    )

threshold = chosen["threshold"]

print("\nChosen threshold:", threshold)
print("Training CV threshold metrics:")
print(json.dumps(chosen, indent=2))

# Train once on the full training split.
selected_pipeline.fit(X_train, y_train)

test_probabilities = selected_pipeline.predict_proba(X_test)[:, 1]
test_predictions = (
    test_probabilities >= threshold
).astype(int)

dummy = DummyClassifier(strategy="prior")
dummy.fit(X_train, y_train)

dummy_probabilities = dummy.predict_proba(X_test)[:, 1]

test_results = {
    "model": best_name,
    "threshold": threshold,
    "roc_auc": float(
        roc_auc_score(y_test, test_probabilities)
    ),
    "average_precision": float(
        average_precision_score(y_test, test_probabilities)
    ),
    "accuracy": float(
        accuracy_score(y_test, test_predictions)
    ),
    "balanced_accuracy": float(
        balanced_accuracy_score(y_test, test_predictions)
    ),
    "precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, test_predictions)
    ),
    "confusion_matrix": confusion_matrix(
        y_test, test_predictions
    ).tolist(),
    "dummy_roc_auc": float(
        roc_auc_score(y_test, dummy_probabilities)
    ),
    "dummy_average_precision": float(
        average_precision_score(y_test, dummy_probabilities)
    ),
}

print("\nFINAL TEST RESULTS")
print(json.dumps(test_results, indent=2))

joblib.dump(
    {
        "model": selected_pipeline,
        "features": features,
        "threshold": threshold,
        "target": "serum_vitamin_d_below_50_nmol_l",
        "research_only": True,
    },
    MODELS / "vitamin_d_cv_research_model.joblib"
)

with open(
    REPORTS / "vitamin_d_cv_results.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        {
            "cross_validation": cv_results,
            "threshold_search": threshold_results,
            "final_test": test_results,
        },
        file,
        indent=2
    )

print("\nSaved model and report successfully.")
