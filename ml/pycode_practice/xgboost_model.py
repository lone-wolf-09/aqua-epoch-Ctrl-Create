import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier


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

X = df.drop(
    "Potability",
    axis=1
)

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


# Remove invalid targets

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
# 4. IMPUTATION
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X.columns,
    index=X_train.index
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X.columns,
    index=X_test.index
)


# ============================================================
# 5. CLASS IMBALANCE
# ============================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nClass counts:")
print("Negative:", negative)
print("Positive:", positive)

print("\nscale_pos_weight:")
print(scale_pos_weight)


# ============================================================
# 6. XGBOOST MODEL
# ============================================================

model = XGBClassifier(

    n_estimators=400,

    max_depth=5,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    min_child_weight=3,

    reg_lambda=1,

    scale_pos_weight=scale_pos_weight,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 7. TRAIN
# ============================================================

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# 8. PROBABILITIES
# ============================================================

y_proba = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 9. DEFAULT THRESHOLD
# ============================================================

y_pred = (
    y_proba >= 0.50
).astype(int)


print("\n" + "=" * 70)
print("XGBOOST @ THRESHOLD 0.50")
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

for threshold in np.arange(
    0.10,
    0.71,
    0.05
):

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