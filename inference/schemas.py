from pydantic import BaseModel
from typing import Optional

SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}


class PredictionResult(BaseModel):
    sentiment: str
    label: int
    confidence: float
    model_used: str
