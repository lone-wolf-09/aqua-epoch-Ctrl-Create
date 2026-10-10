from fastapi import APIRouter, HTTPException

from app.schemas.prediction import (
    WaterQualityInput,
    PredictionResponse
)

from app.services.model_service import model_service


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)


@router.post(
    "",
    response_model=PredictionResponse
)
def predict(data: WaterQualityInput):

    try:

        result = model_service.predict(
            data.model_dump()
        )

        return result

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