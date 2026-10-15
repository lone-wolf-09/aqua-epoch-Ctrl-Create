
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "aqua_epoch_dataset.csv"

df = pd.read_csv(DATA)
df = df.replace(r"^\s*$", np.nan, regex=True)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.loc[~df.isna().all(axis=1)].copy()

df["negative_pair"] = (df["Hardness"] < 0) & (df["Fluoride"] < 0)
df["extreme_pair"] = (df["TDS"] > 100000) & (df["Turbidity"] > 10)
df["invalid_pH"] = (df["pH"] < 0) | (df["pH"] > 14)

groups = {
    "ordinary": ~(df["negative_pair"] | df["extreme_pair"] | df["invalid_pH"]),
    "negative_pair": df["negative_pair"],
    "extreme_pair": df["extreme_pair"],
    "invalid_pH": df["invalid_pH"],
}

for name, mask in groups.items():
    subset = df.loc[mask]
    print(f"\n=== {name.upper()} ===")
    print("Rows:", len(subset))
    print("Target distribution:")
    print(subset["Potability"].value_counts(dropna=False).sort_index())
    print("\nSelected measurements:")
    columns = [
        "pH", "Hardness", "TDS", "Fluoride",
        "Turbidity", "Potability"
    ]
    print(subset[columns].describe().round(3).to_string())

print("\n=== NEGATIVE PAIR SAMPLE ===")
print(
    df.loc[df["negative_pair"],
           ["Hardness", "Fluoride", "Calcium", "Magnesium",
            "TDS", "Potability"]]
    .head(20)
    .to_string(index=False)
)

print("\n=== EXTREME PAIR SAMPLE ===")
print(
    df.loc[df["extreme_pair"],
           ["TDS", "Turbidity", "Hardness", "Fluoride",
            "pH", "Potability"]]
    .sort_values("TDS", ascending=False)
    .head(20)
    .to_string(index=False)
)
