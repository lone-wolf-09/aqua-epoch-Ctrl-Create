
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"

CLASSIFIER_PATH = MODEL_DIR / "official_classifier_baseline.pkl"
REGRESSOR_PATH = MODEL_DIR / "official_ph_regressor_candidate.pkl"

FEATURES = [
    "pH", "Hardness", "TDS", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes",
    "Turbidity", "Calcium", "Magnesium", "Sodium",
    "Potassium", "Chloride", "Nitrate", "Fluoride",
    "Iron", "Alkalinity"
]

REGRESSION_FEATURES = [
    feature for feature in FEATURES if feature != "pH"
]


class ModelService:
    def __init__(self):
        self.classifier = joblib.load(CLASSIFIER_PATH)
        self.regressor = joblib.load(REGRESSOR_PATH)

    def predict(self, data: dict) -> dict:
        input_data = pd.DataFrame(
            [{feature: data[feature] for feature in FEATURES}],
            columns=FEATURES
        )

        # Apply the same impossible-value handling used during training.
        input_data.loc[
            (input_data["pH"] < 0) | (input_data["pH"] > 14), "pH"
        ] = float("nan")
        input_data.loc[input_data["Hardness"] < 0, "Hardness"] = float("nan")
        input_data.loc[input_data["Fluoride"] < 0, "Fluoride"] = float("nan")

        prediction = int(self.classifier.predict(input_data)[0])

        probabilities = self.classifier.predict_proba(input_data)[0]
        class_index = list(self.classifier.classes_).index(prediction)
        confidence = float(probabilities[class_index])

        regression_input = input_data[REGRESSION_FEATURES]
        predicted_ph = float(self.regressor.predict(regression_input)[0])

        return {
            "prediction": prediction,
            "status": "Safe" if prediction == 1 else "Unsafe",
            "confidence": confidence,
            "predicted_pH": predicted_ph
        }


model_service = ModelService()
