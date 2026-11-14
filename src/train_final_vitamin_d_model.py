
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
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
DATA_PATH = BASE / "data" / "processed" / "nutrition_30day_ml.csv"
MODEL_PATH = BASE / "models" / "vitamin_d_final_research_model.joblib"
REPORT_PATH = BASE / "reports" / "vitamin_d_final_research_results.json"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ========================================
# 1. LOAD DATA
# ========================================

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["LBXVIDMS"]).copy()

df["low_vitamin_d"] = (
    df["LBXVIDMS"] < 50
).astype(int)

print("Total records:", len(df))
print("\nTarget distribution:")
print(df["low_vitamin_d"].value_counts())


# ========================================
# 2. PREPARE SUPPLEMENT FEATURES
# ========================================

amount_col = "vitamin_d_supplement_30day_mcg"
count_col = "supplement_count_30day"

# Normalize tiny SAS values representing zero.
for column in [amount_col, count_col]:
    df.loc[
        df[column].notna()
        & (df[column].abs() < 1e-20),
        column
    ] = 0

# Record whether the original amount was absent.
df["vitamin_d_30day_missing"] = (
    df[amount_col].isna().astype(int)
)

# If the person reported zero supplements,
# their supplement vitamin D intake is zero.
zero_supplements = df[count_col].eq(0)

df.loc[
    zero_supplements & df[amount_col].isna(),
    amount_col
] = 0

# Other missing amounts remain missing.
# The pipeline will impute them using training data.


# ========================================
# 3. SELECT FEATURES
# ========================================

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
    "vitamin_d_supplement_30day_mcg",
    "supplement_count_30day",
    "vitamin_d_30day_missing",
]

categorical_features = [
    "sex_code",
    "examination_period",
]

features = numeric_features + categorical_features

X = df[features]
y = df["low_vitamin_d"]


# ========================================
# 4. TRAIN / TEST SPLIT
# ========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ========================================
# 5. BUILD ML PIPELINE
# ========================================

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
    ("categorical", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42,
    )),
])


# ========================================
# 6. CROSS-VALIDATION
# ========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scores = cross_validate(
    model,
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

cv_results = {
    "roc_auc_mean": float(
        scores["test_roc_auc"].mean()
    ),
    "roc_auc_std": float(
        scores["test_roc_auc"].std()
    ),
    "average_precision_mean": float(
        scores["test_average_precision"].mean()
    ),
    "balanced_accuracy_mean": float(
        scores["test_balanced_accuracy"].mean()
    ),
}

print("\nCROSS-VALIDATION RESULTS")
print(json.dumps(cv_results, indent=2))


# ========================================
# 7. SELECT THRESHOLD ON TRAINING DATA
# ========================================

oof_probabilities = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]

threshold_results = []

for threshold in np.arange(0.10, 0.91, 0.05):

    predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "threshold": round(float(threshold), 2),
        "recall": float(
            recall_score(y_train, predictions)
        ),
        "precision": float(
            precision_score(
                y_train,
                predictions,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_train,
                predictions,
            )
        ),
    })

# Research objective: at least 80% recall.
eligible = [
    result for result in threshold_results
    if result["recall"] >= 0.80
]

if eligible:
    selected = max(
        eligible,
        key=lambda result: result["balanced_accuracy"]
    )
else:
    selected = max(
        threshold_results,
        key=lambda result: result["balanced_accuracy"]
    )

threshold = selected["threshold"]

print("\nSELECTED THRESHOLD")
print(json.dumps(selected, indent=2))


# ========================================
# 8. FIT AND EVALUATE
# ========================================

model.fit(X_train, y_train)

probabilities = model.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)

results = {
    "roc_auc": float(
        roc_auc_score(y_test, probabilities)
    ),
    "average_precision": float(
        average_precision_score(
            y_test, probabilities
        )
    ),
    "accuracy": float(
        accuracy_score(y_test, predictions)
    ),
    "balanced_accuracy": float(
        balanced_accuracy_score(
            y_test, predictions
        )
    ),
    "precision": float(
        precision_score(
            y_test,
            predictions,
            zero_division=0,
        )
    ),
    "recall": float(
        recall_score(y_test, predictions)
    ),
    "brier_score": float(
        brier_score_loss(y_test, probabilities)
    ),
    "confusion_matrix": confusion_matrix(
        y_test, predictions
    ).tolist(),
}

print("\nDEVELOPMENT TEST RESULTS")
print(json.dumps(results, indent=2))


# ========================================
# 9. SAVE RESEARCH MODEL
# ========================================

artifact = {
    "model": model,
    "features": features,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "threshold": threshold,
    "target": "serum_25ohd_below_50_nmol_l",
    "research_only": True,
    "supplement_window": "30_day",
    "model_name": "Logistic Regression",
    "zero_supplement_count_means_zero_intake": True,
    "missing_indicator_before_zero_fill": True,
}

joblib.dump(artifact, MODEL_PATH)

report = {
    "model": "Logistic Regression",
    "feature_set": "30 Day + Season",
    "training_records": len(X_train),
    "testing_records": len(X_test),
    "cross_validation": cv_results,
    "threshold_selection": selected,
    "development_test": results,
    "limitations": [
        "NHANES general population, not astronauts",
        "Cross-sectional associations, not future risk",
        "Previously examined test partition",
        "Survey weights not incorporated",
        "Supplement missing-value coding needs final verification",
        "No independent external validation",
    ],
}

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(report, file, indent=2)

print("\nMODEL SAVED SUCCESSFULLY")
print("Model:", MODEL_PATH)
print("Report:", REPORT_PATH)
print("\nResearch use only.")
