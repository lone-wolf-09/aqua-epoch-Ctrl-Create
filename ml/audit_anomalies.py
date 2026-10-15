
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "aqua_epoch_dataset.csv"
OUT = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df = df.replace(r"^\s*$", np.nan, regex=True)

features = [c for c in df.columns if c != "Potability"]

for col in features + ["Potability"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

# Separate export artifacts from actual records.
empty_mask = df.isna().all(axis=1)
records = df.loc[~empty_mask].copy()

print("Original shape:", df.shape)
print("Completely empty rows removed from working copy:", empty_mask.sum())
print("Non-empty records:", len(records))
print("Duplicate non-empty records:", records.duplicated().sum())

print("\nMissingness among non-empty records:")
missing = pd.DataFrame({
    "missing_count": records.isna().sum(),
    "missing_percent": (records.isna().mean() * 100).round(3),
})
print(missing.to_string())
missing.to_csv(OUT / "nonempty_missingness.csv")

# Record flags for review, without dropping any records.
flags = pd.DataFrame(index=records.index)
flags["pH_outside_0_14"] = (
    (records["pH"] < 0) | (records["pH"] > 14)
)
flags["negative_Hardness"] = records["Hardness"] < 0
flags["negative_Fluoride"] = records["Fluoride"] < 0
flags["TDS_above_100000"] = records["TDS"] > 100000
flags["Turbidity_above_10"] = records["Turbidity"] > 10

flagged = records.loc[flags.any(axis=1)].copy()
flagged["anomaly_flags"] = flags.loc[flags.any(axis=1)].apply(
    lambda row: ", ".join(row.index[row.fillna(False)].tolist()),
    axis=1,
)
flagged.to_csv(OUT / "flagged_records.csv", index=False)

print("\nFlagged records:", len(flagged))
print("Flag counts:")
print(flags.sum().sort_values(ascending=False).to_string())

print("\nTDS distribution:")
print(records["TDS"].describe(percentiles=[.90, .95, .99, .999]))

print("\nTarget counts after excluding empty rows:")
print(records["Potability"].value_counts(dropna=False).sort_index())

print("\nReports written to:", OUT)
