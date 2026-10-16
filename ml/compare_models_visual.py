
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    classification_report, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, recall_score, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
FEATURE_CANDIDATES = [
    "TDS", "Hardness", "Chloramines", "Sulfate", "Conductivity",
    "Organic_carbon", "Trihalomethanes", "Turbidity", "Calcium",
    "Magnesium", "Sodium", "Potassium", "Chloride", "Nitrate",
    "Fluoride", "Iron", "Alkalinity", "pH",
]
CLASS_TARGETS = ["Potability", "potability", "target", "Target", "label", "Label"]
PH_TARGETS = ["pH", "PH", "ph"]


def args_parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None, help="Dataset CSV path")
    p.add_argument("--output", default="ml/reports/model_comparison", help="Output directory")
    return p.parse_args()


def find_data(root, supplied):
    if supplied:
        path = Path(supplied)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path
    for rel in [
        "ml/data/official/aqua_epoch_dataset.csv",
        "ml/data/aqua_epoch_dataset.csv",
        "ml/data/water_potability_practice.csv",
        "ml/data/water_potability.csv",
    ]:
        path = root / rel
        if path.exists():
            return path
    raise FileNotFoundError("Dataset not found. Pass --data path/to/dataset.csv")


def existing(cols, options):
    return next((x for x in options if x in cols), None)


def features_for(df, target, excluded=()):
    cols = [x for x in FEATURE_CANDIDATES if x in df.columns and x != target and x not in excluded]
    if not cols:
        cols = [x for x in df.select_dtypes(include=np.number).columns
                if x != target and x not in excluded and x not in CLASS_TARGETS]
    return cols


def pipeline(model, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def correlation_plot(df, cols, target, out):
    names = [c for c in cols if c in df.columns]
    if target in df.columns:
        names.append(target)
    corr = df[names].apply(pd.to_numeric, errors="coerce").corr()
    plt.figure(figsize=(max(8, len(names)*0.62), max(6, len(names)*0.52)))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=len(names) <= 10, fmt=".2f", linewidths=.3)
    plt.title("Numeric feature correlation heatmap")
    save_plot(out / "correlation_heatmap.png")
    corr.to_csv(out / "correlation_matrix.csv")


def classifier_comparison(df, features, target, out):
    d = df[features + [target]].copy()
    d[target] = pd.to_numeric(d[target], errors="coerce")
    d = d.dropna(subset=[target])
    d = d[d[target].isin([0, 1])]
    if len(d) < 20 or d[target].nunique() != 2:
        print("[SKIP] Classification needs at least 20 labelled rows and both classes.")
        return
    X, y = d[features], d[target].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.20, random_state=SEED, stratify=y)
    models = {
        "Logistic Regression": pipeline(LogisticRegression(max_iter=2000, class_weight="balanced"), True),
        "Random Forest": pipeline(RandomForestClassifier(n_estimators=250, min_samples_leaf=2,
            class_weight="balanced", random_state=SEED, n_jobs=-1)),
        "Extra Trees": pipeline(ExtraTreesClassifier(n_estimators=250, min_samples_leaf=2,
            class_weight="balanced", random_state=SEED, n_jobs=-1)),
        "Gradient Boosting": pipeline(GradientBoostingClassifier(random_state=SEED)),
    }
    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        try:
            score = model.predict_proba(Xte)[:, 1]
            auc = roc_auc_score(yte, score)
            ap = average_precision_score(yte, score)
        except Exception:
            auc, ap = np.nan, np.nan
        rows.append({"model": name, "accuracy": accuracy_score(yte, pred),
            "precision_class_1": precision_score(yte, pred, zero_division=0),
            "recall_class_1": recall_score(yte, pred, zero_division=0),
            "f1_class_1": f1_score(yte, pred, zero_division=0),
            "roc_auc": auc, "average_precision": ap})
        fitted[name] = (model, pred)
    result = pd.DataFrame(rows).sort_values("f1_class_1", ascending=False)
    result.to_csv(out / "classification_metrics.csv", index=False)
    print("\n=== CLASSIFICATION (20% held-out test split) ===\n", result.round(4).to_string(index=False))

    plotdata = result.set_index("model")[["accuracy", "precision_class_1", "recall_class_1", "f1_class_1"]]
    ax = plotdata.plot(kind="bar", figsize=(11, 6), ylim=(0, 1.05))
    ax.set_title("Classifier comparison"); ax.set_ylabel("Score"); ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    save_plot(out / "classification_model_comparison.png")

    best = result.iloc[0]["model"]
    model, pred = fitted[best]
    cm = confusion_matrix(yte, pred, labels=[0, 1])
    pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]).to_csv(
        out / "best_classifier_confusion_matrix.csv")
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"], yticklabels=["Actual 0", "Actual 1"])
    plt.title(f"Confusion matrix — {best}"); plt.ylabel("Actual"); plt.xlabel("Predicted")
    save_plot(out / "best_classifier_confusion_matrix.png")
    with open(out / "best_classifier_report.json", "w", encoding="utf-8") as f:
        json.dump({"best_model_by_f1": best,
                   "report": classification_report(yte, pred, output_dict=True, zero_division=0)}, f, indent=2)

    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        imp = pd.Series(estimator.feature_importances_, index=features).sort_values()
        imp.to_csv(out / "classifier_feature_importance.csv", header=["importance"])
        plt.figure(figsize=(8, max(4, len(imp)*.34)))
        sns.barplot(x=imp.values, y=imp.index)
        plt.title(f"Feature importance — {best}"); plt.xlabel("Model-reported importance")
        save_plot(out / "classifier_feature_importance.png")
    print(f"Best classifier by F1 on this split: {best}")
    print("Reminder: results are dataset-specific, not proof of real-world water safety.")


def regressor_comparison(df, features, target, out):
    d = df[features + [target]].copy()
    d[target] = pd.to_numeric(d[target], errors="coerce")
    d = d.dropna(subset=[target])
    d = d[(d[target] >= 0) & (d[target] <= 14)]
    if len(d) < 30:
        print(f"[SKIP] Regression needs at least 30 valid pH rows; found {len(d)}.")
        return
    X, y = d[features], d[target].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.20, random_state=SEED)
    models = {
        "Median baseline": pipeline(DummyRegressor(strategy="median")),
        "Random Forest": pipeline(RandomForestRegressor(n_estimators=250, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1)),
        "Extra Trees": pipeline(ExtraTreesRegressor(n_estimators=250, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1)),
        "Gradient Boosting": pipeline(GradientBoostingRegressor(random_state=SEED)),
    }
    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rows.append({"model": name, "MAE": mean_absolute_error(yte, pred),
                     "RMSE": np.sqrt(mean_squared_error(yte, pred)), "R2": r2_score(yte, pred)})
        fitted[name] = (model, pred)
    result = pd.DataFrame(rows).sort_values("MAE")
    result.to_csv(out / "regression_metrics.csv", index=False)
    print("\n=== REGRESSION (20% held-out test split) ===\n", result.round(4).to_string(index=False))
    ax = result.set_index("model")[["MAE", "RMSE"]].plot(kind="bar", figsize=(10, 5))
    ax.set_title("Regressor error comparison (lower is better)"); ax.set_ylabel("Error in pH units")
    ax.set_xlabel(""); plt.xticks(rotation=20, ha="right")
    save_plot(out / "regression_error_comparison.png")
    ax = result.set_index("model")[["R2"]].plot(kind="bar", figsize=(8, 4))
    ax.axhline(0, linewidth=1); ax.set_title("R² comparison (higher is better)"); ax.set_ylabel("R²")
    ax.set_xlabel(""); plt.xticks(rotation=20, ha="right")
    save_plot(out / "regression_r2_comparison.png")

    best = result.iloc[0]["model"]
    model, pred = fitted[best]
    pd.DataFrame({"actual_pH": yte.to_numpy(), "predicted_pH": pred}).to_csv(
        out / "best_regressor_predictions.csv", index=False)
    plt.figure(figsize=(6, 6))
    plt.scatter(yte, pred, alpha=.45)
    low = min(float(yte.min()), float(np.min(pred))); high = max(float(yte.max()), float(np.max(pred)))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.title(f"Actual vs predicted pH — {best}"); plt.xlabel("Actual pH"); plt.ylabel("Predicted pH")
    save_plot(out / "regression_actual_vs_predicted.png")
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        imp = pd.Series(estimator.feature_importances_, index=features).sort_values()
        imp.to_csv(out / "regressor_feature_importance.csv", header=["importance"])
        plt.figure(figsize=(8, max(4, len(imp)*.34)))
        sns.barplot(x=imp.values, y=imp.index)
        plt.title(f"Feature importance for pH — {best}"); plt.xlabel("Model-reported importance")
        save_plot(out / "regressor_feature_importance.png")
    print(f"Best regressor by MAE on this split: {best}")
    print("Compare to the median baseline; low R² means weak explanatory/predictive performance.")


def main():
    args = args_parse()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    data_path = find_data(root, args.data)
    print(f"Dataset: {data_path}")
    df = pd.read_csv(data_path).dropna(how="all").drop_duplicates()
    print(f"Rows/columns after basic cleanup: {df.shape}")
    print("Available columns:", ", ".join(map(str, df.columns)))
    class_target = existing(df.columns, CLASS_TARGETS)
    ph_target = existing(df.columns, PH_TARGETS)
    if class_target:
        feats = features_for(df, class_target)
        if not feats:
            raise ValueError("No numeric classification features found.")
        print(f"\nClassification target: {class_target}\nFeatures ({len(feats)}): {', '.join(feats)}")
        correlation_plot(df, feats, class_target, out)
        classifier_comparison(df, feats, class_target, out)
    else:
        print("[SKIP] Potability target column not found; classification skipped.")
    if ph_target:
        feats = features_for(df, ph_target, excluded=CLASS_TARGETS)
        feats = [x for x in feats if x != ph_target]
        if not feats:
            raise ValueError("No numeric pH regression features found.")
        print(f"\nRegression target: {ph_target}\nFeatures ({len(feats)}): {', '.join(feats)}")
        regressor_comparison(df, feats, ph_target, out)
    else:
        print("[SKIP] pH target column not found; regression skipped.")
    summary = {"dataset": str(data_path), "rows_after_basic_cleanup": int(len(df)),
        "classification_target": class_target, "regression_target": ph_target,
        "seed": SEED, "test_fraction": 0.20,
        "notes": ["Compact feature subset is a starting point, not proven optimal.",
                  "Classification labels are used as stored; verify their meaning.",
                  "pH target rows are restricted to numeric values in [0, 14].",
                  "This quick comparison is not a replacement for the official training pipeline."]}
    with open(out / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFinished. Results and plots saved in: {out}")


if __name__ == "__main__":
    main()
