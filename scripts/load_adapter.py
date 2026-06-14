"""Load fine-tuned LoRA adapter on top of base model."""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

BASE = "cardiffnlp/twitter-roberta-base-sentiment"
ADAPTER = "/tmp/twitter_v39_out/sentiment_analyzer/artifacts/models/twitter"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
base = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=3)
model = PeftModel.from_pretrained(base, ADAPTER)

print(f"Trainable params: {model.num_parameters(only_trainable=True):,}")
print(f"Total params: {model.num_parameters():,}")

# Test inference
inputs = tokenizer("This is amazing!", return_tensors="pt")
outputs = model(**inputs)
import torch
pred = torch.argmax(outputs.logits, dim=-1).item()
labels = {0: "negative", 1: "neutral", 2: "positive"}
print(f"Prediction: {labels[pred]}")
