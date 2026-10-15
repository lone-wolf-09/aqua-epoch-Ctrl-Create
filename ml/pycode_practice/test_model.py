import pandas as pd
import joblib


MODEL_PATH = "models/water_quality_pipeline.pkl"

pipeline = joblib.load(MODEL_PATH)

sample = pd.DataFrame([{
    "ph": 7.0,
    "Hardness": 200.0,
    "Solids": 15000.0,
    "Chloramines": 7.0,
    "Sulfate": 300.0,
    "Conductivity": 400.0,
    "Organic_carbon": 15.0,
    "Trihalomethanes": 60.0,
    "Turbidity": 4.0
}])


prediction = pipeline.predict(sample)[0]

probability = pipeline.predict_proba(sample)[0]

print("Prediction:", prediction)
print("Probability:", probability)

if prediction == 1:
    print("Water Status: SAFE")
else:
    print("Water Status: UNSAFE")