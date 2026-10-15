
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

flags = pd.DataFrame(index=df.index)
flags["invalid_pH"] = (df["pH"] < 0) | (df["pH"] > 14)
flags["negative_Hardness"] = df["Hardness"] < 0
flags["negative_Fluoride"] = df["Fluoride"] < 0
flags["extreme_TDS"] = df["TDS"] > 100000
flags["high_Turbidity"] = df["Turbidity"] > 10

flags = flags.fillna(False)

print("Number of rows by number of flags:")
print(flags.sum(axis=1).value_counts().sort_index())

print("\nOverlap between flags:")
print(flags.astype(int).T.dot(flags.astype(int)))

print("\nFlagged rows by target:")
print(
    df.loc[flags.any(axis=1), "Potability"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nHardness examples:")
print(df.loc[flags["negative_Hardness"]].head(10).to_string(index=False))

print("\nFluoride examples:")
print(df.loc[flags["negative_Fluoride"]].head(10).to_string(index=False))

print("\nExtreme TDS examples:")
print(
    df.loc[flags["extreme_TDS"],
           ["TDS", "Hardness", "Fluoride", "Turbidity", "pH", "Potability"]]
    .sort_values("TDS", ascending=False)
    .head(10)
    .to_string(index=False)
)

