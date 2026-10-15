
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "aqua_epoch_dataset.csv"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

expected_columns = {
    "pH", "Hardness", "TDS", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes",
    "Turbidity", "Calcium", "Magnesium", "Sodium",
    "Potassium", "Chloride", "Nitrate", "Fluoride",
    "Iron", "Alkalinity", "Potability"
}

missing_columns = expected_columns - set(df.columns)
extra_columns = set(df.columns) - expected_columns

if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

# Convert feature columns to numeric; invalid strings become missing.
features = sorted(expected_columns - {"Potability"})
for column in features + ["Potability"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

print("\n=== DATASET OVERVIEW ===")
print("Shape:", df.shape)
print("Unexpected columns:", sorted(extra_columns))
print("Completely empty rows:", df.isna().all(axis=1).sum())
print("Exact duplicate rows:", df.duplicated().sum())

print("\n=== MISSING VALUES ===")
missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": (df.isna().mean() * 100).round(2)
})
print(missing.to_string())
missing.to_csv(REPORT_DIR / "missing_values.csv")

# Inspect target values before any training.
print("\n=== POTABILITY TARGET ===")
print(df["Potability"].value_counts(dropna=False).sort_index())

invalid_target = (
    df["Potability"].notna()
    & ~df["Potability"].isin([0, 1])
)
print("Invalid non-binary targets:", int(invalid_target.sum()))

labelled = df[df["Potability"].isin([0, 1])].copy()
print("Rows with valid classification labels:", len(labelled))

regression_rows = df[df["pH"].notna()].copy()
print("Rows with observed pH target:", len(regression_rows))

print("\n=== FEATURE SUMMARY ===")
summary = df[features].describe().T
summary["missing_count"] = df[features].isna().sum()
summary.to_csv(REPORT_DIR / "feature_summary.csv")
print(summary.round(3).to_string())

# Flag suspicious values for investigation, not automatic deletion.
checks = {
    "pH_below_0": df["pH"] < 0,
    "pH_above_14": df["pH"] > 14,
    "negative_Hardness": df["Hardness"] < 0,
    "negative_TDS": df["TDS"] < 0,
    "negative_Fluoride": df["Fluoride"] < 0,
    "TDS_above_100000": df["TDS"] > 100000,
    "Turbidity_above_10": df["Turbidity"] > 10,
}

print("\n=== SUSPICIOUS VALUE FLAGS ===")
for name, mask in checks.items():
    print(f"{name}: {int(mask.fillna(False).sum())}")

# Keep the non-empty, unlabeled records available for inspection.
unlabeled = df[df["Potability"].isna() & df.notna().any(axis=1)]
unlabeled.to_csv(REPORT_DIR / "unlabeled_records.csv", index=False)

print("\nReports saved to:", REPORT_DIR)
