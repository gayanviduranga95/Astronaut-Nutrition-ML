
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed" / "nutrition_merged.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

# Research proxy: reported >=5% weight loss.
# This is not a malnutrition diagnosis.
df["target"] = np.where(
    df["weight_change_pct"].isna(),
    np.nan,
    (df["weight_change_pct"] <= -5).astype(int)
)

# Exclude records with unknown target.
df = df.dropna(subset=["target"]).copy()

features = [
    "age_years",
    "sex_code",
    "bmi",
    "avg_energy_kcal",
    "avg_protein_g",
    "avg_carbohydrate_g",
    "avg_fat_g",
    "avg_calcium_mg",
    "avg_iron_mg",
    "avg_vitamin_d_mcg"
]

X = df[features]
y = df["target"].astype(int)

print("Training records:", len(df))
print("\nTarget distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\n=== MODEL RESULTS ===")
print("Accuracy:", round(
    accuracy_score(y_test, predictions), 4
))
print("Balanced accuracy:", round(
    balanced_accuracy_score(y_test, predictions), 4
))
print("ROC-AUC:", round(
    roc_auc_score(y_test, probabilities), 4
))

print("\nClassification report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(
    y_test, predictions
))

print("\nMajority-class baseline accuracy:",
      round(max(y_test.mean(), 1 - y_test.mean()), 4))

artifact = {
    "model": model,
    "features": features,
    "target_description": (
        "Cross-sectional association with self-reported "
        "weight loss >=5% over the previous year; "
        "not validated malnutrition risk."
    )
}

output = MODEL_DIR / "nutrition_weight_loss_baseline.joblib"
joblib.dump(artifact, output)

print("\nModel saved to:", output)
print("Training completed!")
