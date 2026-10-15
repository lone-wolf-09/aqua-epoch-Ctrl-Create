"""
One-pass pH regression comparison for Aqua-Epoch.
Copy this file to ml/ and run from project root:
    python .\ml\compare_ph_models.py

Compares the existing median baseline and Random Forest with Extra Trees
and Gradient Boosting. Uses the same fixed 70/15/15 split as the baseline.
Select the best model using validation MAE; test metrics are reported for
final comparison only.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ALL_FEATURES = [
    "pH", "Hardness", "TDS", "Chloramines", "Sulfate", "Conductivity",
    "Organic_carbon", "Trihalomethanes", "Turbidity", "Calcium", "Magnesium",
    "Sodium", "Potassium", "Chloride", "Nitrate", "Fluoride", "Iron", "Alkalinity"
]
FEATURES = [c for c in ALL_FEATURES if c != "pH"]
SEED = 42

def score(y_true, pred):
    return {
        "MAE": round(float(mean_absolute_error(y_true, pred)), 6),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, pred))), 6),
        "R2": round(float(r2_score(y_true, pred)), 6),
    }

def main():
    root = Path(__file__).resolve().parent
    data_path = root / "data" / "official" / "aqua_epoch_dataset.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {data_path}. Copy this script into ml/.")

    df = pd.read_csv(data_path).dropna(how="all")
    df["pH"] = pd.to_numeric(df["pH"], errors="coerce")
    df = df[df["pH"].between(0, 14, inclusive="both")].copy()
    y = df["pH"].astype(float)
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X.loc[X["Hardness"] < 0, "Hardness"] = np.nan
    X.loc[X["Fluoride"] < 0, "Fluoride"] = np.nan

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED
    )

    models = {
        "median_dummy": DummyRegressor(strategy="median"),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=2,
            loss="huber", random_state=SEED
        ),
    }

    results = {}
    fitted = {}
    for name, estimator in models.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", estimator)
        ])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results[name] = {
            "validation": score(y_val, pipe.predict(X_val)),
            "test": score(y_test, pipe.predict(X_test))
        }
        print(f"{name:20s} | val MAE {results[name]['validation']['MAE']:.6f} "
              f"| test MAE {results[name]['test']['MAE']:.6f} "
              f"| test R2 {results[name]['test']['R2']:.6f}")

    # Model selection is based ONLY on validation MAE.
    winner = min(results, key=lambda name: results[name]["validation"]["MAE"])
    # Retrain selected model on train + validation; test remains held out.
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)
    selected_estimator = models[winner]
    final_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", selected_estimator)
    ])
    final_model.fit(X_trainval, y_trainval)
    final_test = score(y_test, final_model.predict(X_test))

    report = {
        "valid_pH_rows": int(len(df)),
        "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "selection_rule": "Lowest validation MAE; test set not used for selection",
        "models": results,
        "selected_by_validation_MAE": winner,
        "selected_model_test_metrics_after_train_plus_validation_fit": final_test,
        "features": FEATURES,
        "leakage_check": "pH and Potability excluded from input features"
    }
    (root / "reports").mkdir(exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    (root / "reports" / "ph_model_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    joblib.dump(final_model, root / "models" / "official_ph_regressor_candidate.pkl")
    print("\nSelected by validation MAE:", winner)
    print("Final selected model test metrics:", final_test)
    print("Saved report: ml/reports/ph_model_comparison.json")
    print("Saved candidate: ml/models/official_ph_regressor_candidate.pkl")

if __name__ == "__main__":
    main()
