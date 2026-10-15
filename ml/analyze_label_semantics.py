from pathlib import Path
import argparse
import pandas as pd

FEATURES=["pH","Hardness","TDS","Chloramines","Sulfate","Conductivity","Organic_carbon","Trihalomethanes","Turbidity","Calcium","Magnesium","Sodium","Potassium","Chloride","Nitrate","Fluoride","Iron","Alkalinity"]
RANGES={"pH":(6.5,8.5),"TDS":(0,500),"Turbidity":(0,1),"Iron":(0,.3),"Fluoride":(0,1.5),"Nitrate":(0,50)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",default=None); a=p.parse_args()
    root=Path(__file__).resolve().parent
    path=Path(a.data) if a.data else root/"data"/"official"/"aqua_epoch_dataset.csv"
    d=pd.read_csv(path).dropna(how="all")
    d["Potability"]=pd.to_numeric(d["Potability"],errors="coerce")
    d=d[d.Potability.isin([0,1])]
    print("WARNING: descriptive checks only; these cannot decode the intended class meaning.")
    print("\nClass counts:\n",d.Potability.value_counts().sort_index())
    print("\nFeature medians by class:\n",d.groupby("Potability")[FEATURES].median().T)
    for col,(lo,hi) in RANGES.items():
        x=pd.to_numeric(d[col],errors="coerce")
        inside=x.between(lo,hi)
        print(f"\n{col} in illustrative range [{lo}, {hi}] by class:")
        print(inside.groupby(d.Potability).mean())
    print("\nConclusion: numeric patterns cannot prove class 0/1 means SAFE/UNSAFE.")
    print("Keep the UI labels as 'Class 0' and 'Class 1' until organizers confirm semantics.")
if __name__=="__main__": main()
