from fastapi import FastAPI
from pydantic import BaseModel

from .model import predict
from .monitoring import log_bad_prediction

app = FastAPI()


class FeedbackInput(BaseModel):
    text: str
    prediction: str


class FeedbackOutput(BaseModel):
    status: str


class PredictInput(BaseModel):
    text: str


class PredictOutput(BaseModel):
    sentiment: str
    confidence: float


@app.post("/feedback")
def get_feedback(input: FeedbackInput):
    log_bad_prediction(input.text, input.prediction)
    return FeedbackOutput(status="ok")


@app.post("/predict")
def get_prediction(input: PredictInput):
    label, proba = predict(input.text)
    return PredictOutput(sentiment=label, confidence=proba)
