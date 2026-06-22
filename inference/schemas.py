from pydantic import BaseModel
from typing import Optional

# Label index to sentiment string mapping
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}


class PredictionResult(BaseModel):
    """Result of a single text classification with sentiment, label, confidence and model source."""
    sentiment: str
    label: int
    confidence: float
    model_used: str
