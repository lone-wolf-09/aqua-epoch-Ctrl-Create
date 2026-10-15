from pathlib import Path
import argparse, json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ALL=["pH","Hardness","TDS","Chloramines","Sulfate","Conductivity","Organic_carbon","Trihalomethanes","Turbidity","Calcium","Magnesium","Sodium","Potassium","Chloride","Nitrate","Fluoride","Iron","Alkalinity"]
FEATURES=[c for c in ALL if c!="pH"] # Potability is also excluded: no target leakage
SEED=42

def score(y,p):
    return {"MAE":float(mean_absolute_error(y,p)),"RMSE":float(np.sqrt(mean_squared_error(y,p))),"R2":float(r2_score(y,p))}

def main():
    a=argparse.ArgumentParser(); a.add_argument("--data",default=None); args=a.parse_args()
    root=Path(__file__).resolve().parent
    path=Path(args.data) if args.data else root/"data"/"official"/"aqua_epoch_dataset.csv"
    df=pd.read_csv(path).dropna(how="all")
    df["pH"]=pd.to_numeric(df["pH"],errors="coerce")
    df=df[df["pH"].between(0,14,inclusive="both")].copy()
    y=df["pH"].astype(float)
    X=df[FEATURES].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    X.loc[X.Hardness<0,"Hardness"]=np.nan
    X.loc[X.Fluoride<0,"Fluoride"]=np.nan
    X_train,X_tmp,y_train,y_tmp=train_test_split(X,y,test_size=.30,random_state=SEED)
    X_val,X_test,y_val,y_test=train_test_split(X_tmp,y_tmp,test_size=.50,random_state=SEED)
    models={
      "median_dummy":Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),("model",DummyRegressor(strategy="median"))]),
      "random_forest":Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),("model",RandomForestRegressor(n_estimators=300,min_samples_leaf=2,random_state=SEED,n_jobs=-1))])
    }
    out={"rows_with_valid_pH":len(df),"features":FEATURES,"leakage_check":"pH and Potability excluded from inputs",
         "split_sizes":{"train":len(X_train),"validation":len(X_val),"test":len(X_test)},"models":{}}
    fitted={}
    for name,m in models.items():
        m.fit(X_train,y_train); fitted[name]=m
        out["models"][name]={"validation":score(y_val,m.predict(X_val)),"test":score(y_test,m.predict(X_test))}
    (root/"reports").mkdir(exist_ok=True); (root/"models").mkdir(exist_ok=True)
    joblib.dump(fitted["random_forest"],root/"models"/"official_ph_regressor_baseline.pkl")
    out["saved_model"]="models/official_ph_regressor_baseline.pkl"
    (root/"reports"/"regression_baseline.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
