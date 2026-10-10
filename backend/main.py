
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import pandas as pd
import joblib


# -----------------------------
# Create FastAPI application
# -----------------------------

app = FastAPI(
    title="Water Quality Prediction API",
    description="ML API for predicting water potability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Load trained ML pipeline
# -----------------------------

MODEL_PATH = "models/water_quality_pipeline.pkl"

model = joblib.load(MODEL_PATH)


# -----------------------------
# Request schema
# -----------------------------

from pydantic import BaseModel, Field


class WaterQualityInput(BaseModel):
    ph: float = Field(ge=0, le=14)
    Hardness: float = Field(ge=0)
    Solids: float = Field(ge=0)
    Chloramines: float = Field(ge=0)
    Sulfate: float = Field(ge=0)
    Conductivity: float = Field(ge=0)
    Organic_carbon: float = Field(ge=0)
    Trihalomethanes: float = Field(ge=0)
    Turbidity: float = Field(ge=0)



class PredictionResponse(BaseModel):
    prediction: int
    status: str
    confidence: float

# -----------------------------
# Root endpoint
# -----------------------------

@app.get("/")
def root():
    return {
        "message": "Water Quality Prediction API is running"
    }


# -----------------------------
# Prediction endpoint
# -----------------------------


from fastapi import HTTPException
@app.post("/predict", response_model=PredictionResponse)
def predict(data: WaterQualityInput):

    try:
        input_data = pd.DataFrame([data.model_dump()])

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        confidence = float(max(probabilities))

        status = "SAFE" if prediction == 1 else "UNSAFE"

        return {
            "prediction": int(prediction),
            "status": status,
            "confidence": confidence
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction service encountered an internal error."
        )



@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }