from pathlib import Path
import argparse, json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, average_precision_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

FEATURES = ["pH","Hardness","TDS","Chloramines","Sulfate","Conductivity","Organic_carbon","Trihalomethanes","Turbidity","Calcium","Magnesium","Sodium","Potassium","Chloride","Nitrate","Fluoride","Iron","Alkalinity"]
SEED = 42

def metrics(y, pred, prob):
    return {"accuracy":float(accuracy_score(y,pred)), "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "precision_class_1":float(precision_score(y,pred,zero_division=0)),
            "recall_class_1":float(recall_score(y,pred,zero_division=0)),
            "f1_class_1":float(f1_score(y,pred,zero_division=0)),
            "average_precision_class_1":float(average_precision_score(y,prob)),
            "roc_auc":float(roc_auc_score(y,prob)),
            "confusion_matrix_labels_0_1":confusion_matrix(y,pred,labels=[0,1]).tolist()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default=None)
    args=p.parse_args()
    root=Path(__file__).resolve().parent
    path=Path(args.data) if args.data else root/"data"/"official"/"aqua_epoch_dataset.csv"
    df=pd.read_csv(path).dropna(how="all")
    df["Potability"]=pd.to_numeric(df["Potability"],errors="coerce")
    df=df[df["Potability"].isin([0,1])].copy()
    y=df["Potability"].astype(int)
    X=df[FEATURES].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    X.loc[(X.pH<0)|(X.pH>14),"pH"]=np.nan
    X.loc[X.Hardness<0,"Hardness"]=np.nan
    X.loc[X.Fluoride<0,"Fluoride"]=np.nan
    X_train,X_tmp,y_train,y_tmp=train_test_split(X,y,test_size=.30,random_state=SEED,stratify=y)
    X_val,X_test,y_val,y_test=train_test_split(X_tmp,y_tmp,test_size=.50,random_state=SEED,stratify=y_tmp)
    models={
      "majority_dummy":Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),("model",DummyClassifier(strategy="most_frequent"))]),
      "random_forest":Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),("model",RandomForestClassifier(n_estimators=300,min_samples_leaf=2,random_state=SEED,n_jobs=-1))])
    }
    out={"rows":len(df),"class_counts":{str(k):int(v) for k,v in y.value_counts().sort_index().items()},
         "class_semantics":"UNKNOWN: do not map to SAFE/UNSAFE without authoritative documentation",
         "split_sizes":{"train":len(X_train),"validation":len(X_val),"test":len(X_test)},"models":{}}
    fitted={}
    for name,m in models.items():
        m.fit(X_train,y_train); fitted[name]=m
        ix=list(m.classes_).index(1)
        vp=m.predict_proba(X_val)[:,ix]; tp=m.predict_proba(X_test)[:,ix]
        out["models"][name]={"validation":metrics(y_val,(vp>=.5).astype(int),vp),
                             "test":metrics(y_test,(tp>=.5).astype(int),tp)}
    (root/"reports").mkdir(exist_ok=True); (root/"models").mkdir(exist_ok=True)
    joblib.dump(fitted["random_forest"],root/"models"/"official_classifier_baseline.pkl")
    out["saved_model"]="models/official_classifier_baseline.pkl"
    (root/"reports"/"classification_baseline.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
