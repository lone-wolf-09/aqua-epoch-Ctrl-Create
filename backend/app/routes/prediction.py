
from fastapi import APIRouter, HTTPException

from app.schemas.prediction import (
    WaterQualityInput,
    PredictionResponse,
)
from app.services.model_service import model_service

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


@router.post("/", response_model=PredictionResponse)
def predict(data: WaterQualityInput):
    try:
        return model_service.predict(data.model_dump())
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the backend logs.",
        ) from error
