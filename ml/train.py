import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# -----------------------------
# 1. Load Dataset
# -----------------------------

DATA_PATH = "data/water_potability.csv"

df = pd.read_csv(DATA_PATH, encoding="cp1252")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())


# -----------------------------
# 2. Separate Features / Target
# -----------------------------

X = df.drop("Potability", axis=1)
y = pd.to_numeric(df["Potability"], errors="coerce")

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

valid_targets = y.isin([0, 1])
removed_targets = int((~valid_targets).sum())
X = X.loc[valid_targets]
y = y.loc[valid_targets].astype(int)

print("\nFeature values after numeric conversion (invalid values become missing):")
print(X.isnull().sum())

if removed_targets:
    print(f"\nRemoved {removed_targets} rows with missing, invalid, or non-binary targets.")

if X.empty or y.nunique() < 2:
    raise ValueError("Training data must contain valid features and both target classes (0 and 1).")

# -----------------------------
# 3. Train / Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# 4. Preprocessing
# -----------------------------

numeric_features = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        )
    ]
)


# -----------------------------
# 5. Model
# -----------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)


# -----------------------------
# 6. Complete ML Pipeline
# -----------------------------

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# -----------------------------
# 7. Train
# -----------------------------

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Training complete.")


# -----------------------------
# 8. Evaluation
# -----------------------------

# Default prediction using threshold = 0.50
y_proba = pipeline.predict_proba(X_test)[:, 1]

y_pred = (y_proba >= 0.50).astype(int)

accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print("DEFAULT THRESHOLD = 0.50")
print("========================================")

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC:")
print(roc_auc_score(y_test, y_proba))

print("\nPR-AUC:")
print(average_precision_score(y_test, y_proba))


# -----------------------------
# Threshold Analysis
# -----------------------------

print("\n========================================")
print("THRESHOLD ANALYSIS")
print("========================================")

thresholds = np.arange(0.10, 0.91, 0.05)

print(
    f"\n{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'Accuracy':<12}"
)

best_threshold = 0.50
best_f1 = 0

for threshold in thresholds:

    predictions = (y_proba >= threshold).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    acc = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.3f}"
        f"{recall:<12.3f}"
        f"{f1:<12.3f}"
        f"{acc:<12.3f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


print("\nBest threshold based on Class 1 F1:")
print(best_threshold)

print("Best Class 1 F1:")
print(best_f1)


# -----------------------------
# 9. Save Complete Pipeline
# -----------------------------

MODEL_PATH = "models/water_quality_pipeline.pkl"

joblib.dump(pipeline, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")