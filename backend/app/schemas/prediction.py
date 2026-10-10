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


