import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/water_potability.csv"

df = pd.read_csv(
    DATA_PATH,
    encoding="cp1252"
)

print("\nDataset shape:")
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


# Keep only valid target rows

valid_targets = y.isin([0, 1])

X = X.loc[valid_targets]
y = y.loc[valid_targets].astype(int)


print("\nClass distribution:")
print(y.value_counts())

print("\nClass percentages:")
print(y.value_counts(normalize=True))


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. PREPROCESSING
# ============================================================

numeric_features = X.columns.tolist()


# Logistic Regression needs scaling.
# Tree models don't really need it,
# but we'll keep the preprocessing consistent for now.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]),
            numeric_features
        )
    ]
)


# ============================================================
# 5. MODELS
# ============================================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
}


# ============================================================
# 6. TRAIN + EVALUATE
# ============================================================

results = []


for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"TRAINING: {name}")
    print("=" * 60)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    # Probability of class 1
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Default threshold
    y_pred = (
        y_proba >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_proba
    )

    pr_auc = average_precision_score(
        y_test,
        y_proba
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })


# ============================================================
# 7. RESULTS
# ============================================================

results_df = pd.DataFrame(results)


print("\n\n")
print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 8. BEST MODEL BY F1
# ============================================================

best_model = results_df.loc[
    results_df["F1"].idxmax()
]

print("\n" + "=" * 80)
print("BEST MODEL BY CLASS 1 F1")
print("=" * 80)

print(best_model)
