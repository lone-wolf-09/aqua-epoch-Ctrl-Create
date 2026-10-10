import joblib
import pandas as pd


MODEL_PATH = "models/water_quality_pipeline.pkl"


class ModelService:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict(self, data: dict):

        input_data = pd.DataFrame([data])

        prediction = self.model.predict(input_data)[0]

        probabilities = self.model.predict_proba(input_data)[0]

        confidence = float(max(probabilities))

        status = "SAFE" if prediction == 1 else "UNSAFE"

        return {
            "prediction": int(prediction),
            "status": status,
            "confidence": confidence
        }


model_service = ModelService()
