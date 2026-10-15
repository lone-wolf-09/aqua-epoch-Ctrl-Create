
from pydantic import BaseModel, Field


class WaterQualityInput(BaseModel):
    pH: float = Field(ge=0, le=14)
    Hardness: float = Field(ge=0)
    TDS: float = Field(ge=0)
    Chloramines: float = Field(ge=0)
    Sulfate: float = Field(ge=0)
    Conductivity: float = Field(ge=0)
    Organic_carbon: float = Field(ge=0)
    Trihalomethanes: float = Field(ge=0)
    Turbidity: float = Field(ge=0)
    Calcium: float = Field(ge=0)
    Magnesium: float = Field(ge=0)
    Sodium: float = Field(ge=0)
    Potassium: float = Field(ge=0)
    Chloride: float = Field(ge=0)
    Nitrate: float = Field(ge=0)
    Fluoride: float = Field(ge=0)
    Iron: float = Field(ge=0)
    Alkalinity: float = Field(ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    status: str
    confidence: float
    predicted_pH: float
