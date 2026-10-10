import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/water_potability.csv"

df = pd.read_csv(
    DATA_PATH,
    encoding="cp1252"
)

print("Dataset shape:")
print(df.shape)


# ============================================================
# 2. CLEAN DATA
# ============================================================

X = df.drop("Potability", axis=1)

y = pd.to_numeric(
    df["Potability"],
    errors="coerce"
)

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

valid_targets = y.isin([0, 1])

X = X.loc[valid_targets]
y = y.loc[valid_targets].astype(int)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. PREPROCESSING
# ============================================================

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


# ============================================================
# 5. RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rf)
])


# ============================================================
# 6. HYPERPARAMETER SEARCH
# ============================================================

param_grid = {

    "model__n_estimators": [
        200,
        300,
        500
    ],

    "model__max_depth": [
        None,
        8,
        12,
        16,
        20
    ],

    "model__min_samples_split": [
        2,
        5,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4,
        8
    ],

    "model__max_features": [
        "sqrt",
        "log2",
        None
    ],

    "model__class_weight": [
        "balanced",
        "balanced_subsample"
    ]
}


# ============================================================
# 7. SEARCH
# ============================================================

print("\nStarting Random Forest hyperparameter search...")

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=25,
    scoring="average_precision",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


search.fit(
    X_train,
    y_train
)


# ============================================================
# 8. BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("BEST PARAMETERS")
print("=" * 70)

print(search.best_params_)

print("\nBest Cross-Validation PR-AUC:")
print(search.best_score_)


best_model = search.best_estimator_


# ============================================================
# 9. TEST EVALUATION
# ============================================================

y_proba = best_model.predict_proba(X_test)[:, 1]

y_pred = (
    y_proba >= 0.50
).astype(int)


print("\n" + "=" * 70)
print("TEST RESULTS @ THRESHOLD 0.50")
print("=" * 70)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nROC-AUC:")
print(
    roc_auc_score(
        y_test,
        y_proba
    )
)

print("\nPR-AUC:")
print(
    average_precision_score(
        y_test,
        y_proba
    )
)


# ============================================================
# 10. THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

print(
    f"\n{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

best_threshold = 0.50
best_f1 = 0

for threshold in np.arange(0.10, 0.71, 0.05):

    predictions = (
        y_proba >= threshold
    ).astype(int)

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

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.3f}"
        f"{recall:<12.3f}"
        f"{f1:<12.3f}"
    )

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold


print("\nBest threshold:")
print(best_threshold)

print("\nBest Class 1 F1:")
print(best_f1)
