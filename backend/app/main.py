from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.prediction import router as prediction_router


app = FastAPI(
    title="Water Quality Prediction API",
    description="ML API for predicting water potability",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(prediction_router)


@app.get("/")
def root():

    return {
        "message": "Water Quality Prediction API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }