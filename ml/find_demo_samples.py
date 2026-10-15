
from pathlib import Path

import joblib
import pandas as pd

FEATURES = [
    "pH", "Hardness", "TDS", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes",
    "Turbidity", "Calcium", "Magnesium", "Sodium",
    "Potassium", "Chloride", "Nitrate", "Fluoride",
    "Iron", "Alkalinity",
]

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "official" / "aqua_epoch_dataset.csv"
MODEL = ROOT / "models" / "official_classifier_baseline.pkl"

df = pd.read_csv(DATA).dropna(how="all")
df = df.dropna(subset=FEATURES + ["Potability"]).copy()
df["Potability"] = pd.to_numeric(df["Potability"], errors="coerce")
df = df[df["Potability"].isin([0, 1])].copy()

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")

# Match the preprocessing rules used during training.
X.loc[~X["pH"].between(0, 14), "pH"] = float("nan")
X.loc[X["Hardness"] < 0, "Hardness"] = float("nan")
X.loc[X["Fluoride"] < 0, "Fluoride"] = float("nan")

model = joblib.load(MODEL)
df["predicted_class"] = model.predict(X)
df["confidence"] = model.predict_proba(X).max(axis=1)

for cls in [0, 1]:
    candidates = df[
        (df["Potability"] == cls)
        & (df["predicted_class"] == cls)
    ].sort_values("confidence", ascending=False)

    if candidates.empty:
        print(f"\nNo correctly predicted example found for Class {cls}")
        continue

    row = candidates.iloc[0]
    print(f"\n{'=' * 15} DEMO CLASS {cls} {'=' * 15}")
    print(f"Actual dataset label: {int(row['Potability'])}")
    print(f"Predicted class:      {int(row['predicted_class'])}")
    print(f"Model confidence:     {row['confidence']:.2%}")
    print("\nPaste these values into the dashboard:")
    print(row[FEATURES].to_json(indent=2))
