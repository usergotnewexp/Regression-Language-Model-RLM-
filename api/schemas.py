from pydantic import BaseModel


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: float
    confidence_interval: list
    uncertainty: float
