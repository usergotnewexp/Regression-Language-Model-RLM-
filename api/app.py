"""
FastAPI server for RLM
TODO: Add implementation
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RLM API")

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: float
    confidence_interval: list
    uncertainty: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # TODO: Load model and make prediction
    raise NotImplementedError("Add implementation")

@app.get("/")
async def root():
    return {"message": "RLM API is running"}

