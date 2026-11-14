
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
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
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# ==========================================
# 1. PATHS
# ==========================================

BASE = Path(__file__).resolve().parent.parent

DATA = (
    BASE / "data" / "processed"
    / "nutrition_enhanced_ml.csv"
)

MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv(DATA)

df = df.dropna(subset=["LBXVIDMS"]).copy()

df["low_vitamin_d"] = (
    df["LBXVIDMS"] < 50
).astype(int)

# Missing supplement vitamin D amount does not
# necessarily mean no supplement was taken.
# Preserve that distinction explicitly.
df["supplement_vitamin_d_missing"] = (
    df["supplement_vitamin_d_mcg"]
    .isna()
    .astype(int)
)

print("Records:", len(df))
print("\nTarget distribution:")
print(df["low_vitamin_d"].value_counts())


# ==========================================
# 3. FEATURE GROUPS
# ==========================================

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
]

enhanced_numeric = base_numeric + [
    "supplement_vitamin_d_mcg",
    "supplement_count",
    "supplement_vitamin_d_missing",
]

enhanced_categorical = base_categorical + [
    "examination_period",
]

feature_sets = {
    "Original": {
        "numeric": base_numeric,
        "categorical": base_categorical,
    },
    "Enhanced": {
        "numeric": enhanced_numeric,
        "categorical": enhanced_categorical,
    },
}


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

# Use identical participants for every model.
# This is the same split used in earlier
# experiments, so it is not a fresh holdout.

train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.20,
    random_state=42,
    stratify=df["low_vitamin_d"],
)

y = df["low_vitamin_d"]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nTraining:", len(train_idx))
print("Testing:", len(test_idx))


# ==========================================
# 5. PIPELINE BUILDER
# ==========================================

def build_pipeline(feature_config, classifier):

    numeric = feature_config["numeric"]
    categorical = feature_config["categorical"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True
        )),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        )),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        (
            "categorical",
            categorical_pipeline,
            categorical
        ),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


# ==========================================
# 6. CANDIDATE MODELS
# ==========================================

def get_models():
    return {
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


# ==========================================
# 7. CROSS-VALIDATION
# ==========================================

cv_results = {}
best_auc = -1
best_config = None

for feature_name, config in feature_sets.items():

    features = (
        config["numeric"]
        + config["categorical"]
    )

    X = df[features]

    X_train = X.iloc[train_idx]

    for model_name, classifier in get_models().items():

        name = f"{feature_name} - {model_name}"

        print("\n" + "=" * 55)
        print(name)
        print("=" * 55)

        pipeline = build_pipeline(
            config,
            classifier
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

        cv_results[name] = metrics

        print(json.dumps(metrics, indent=2))

        if metrics["roc_auc_mean"] > best_auc:
            best_auc = metrics["roc_auc_mean"]
            best_config = (
                feature_name,
                model_name
            )


# ==========================================
# 8. SELECT MODEL
# ==========================================

feature_name, model_name = best_config

print("\nSelected:", feature_name, model_name)

config = feature_sets[feature_name]
features = config["numeric"] + config["categorical"]

X = df[features]
X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

pipeline = build_pipeline(
    config,
    get_models()[model_name]
)


# ==========================================
# 9. SELECT THRESHOLD USING TRAINING DATA
# ==========================================

oof_probabilities = cross_val_predict(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]

threshold_results = []

for threshold in np.arange(0.10, 0.91, 0.05):

    predicted = (
        oof_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "threshold": round(float(threshold), 2),
        "recall": float(
            recall_score(y_train, predicted)
        ),
        "precision": float(
            precision_score(
                y_train,
                predicted,
                zero_division=0
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_train,
                predicted
            )
        ),
    })

# Research screening objective:
# Achieve at least 80% recall, then maximize
# balanced accuracy among eligible thresholds.

eligible = [
    result for result in threshold_results
    if result["recall"] >= 0.80
]

if eligible:
    chosen = max(
        eligible,
        key=lambda x: x["balanced_accuracy"]
    )
else:
    chosen = max(
        threshold_results,
        key=lambda x: x["balanced_accuracy"]
    )

threshold = chosen["threshold"]

print("\nChosen threshold:", threshold)
print("Training threshold metrics:")
print(json.dumps(chosen, indent=2))


# ==========================================
# 10. FINAL TEST EVALUATION
# ==========================================

pipeline.fit(X_train, y_train)

probabilities = pipeline.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)

final_results = {
    "feature_set": feature_name,
    "model": model_name,
    "threshold": threshold,
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
            zero_division=0
        )
    ),
    "recall": float(
        recall_score(y_test, predictions)
    ),
    "confusion_matrix": confusion_matrix(
        y_test, predictions
    ).tolist(),
}

print("\n" + "=" * 55)
print("FINAL TEST RESULTS")
print("=" * 55)

print(json.dumps(final_results, indent=2))


# ==========================================
# 11. SAVE MODEL AND REPORT
# ==========================================

model_path = (
    MODEL_DIR
    / "vitamin_d_enhanced_research_model.joblib"
)

report_path = (
    REPORT_DIR
    / "vitamin_d_enhanced_results.json"
)

joblib.dump(
    {
        "model": pipeline,
        "features": features,
        "threshold": threshold,
        "target": "serum_vitamin_d_below_50_nmol_l",
        "research_only": True,
        "feature_set": feature_name,
        "model_name": model_name,
    },
    model_path
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        {
            "cross_validation": cv_results,
            "threshold_search": threshold_results,
            "final_test": final_results,
        },
        file,
        indent=2
    )

print("\nSUCCESS!")
print("Model saved:", model_path)
print("Report saved:", report_path)
print("\nResearch only — not validated for astronauts.")
