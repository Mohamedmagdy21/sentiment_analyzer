"""Test inference on fine-tuned LoRA adapters.

Usage:
  python scripts/inference.py twitter  "I love this!"
  python scripts/inference.py amazon   "Terrible quality"
  python scripts/inference.py twitter  --interactive
"""
import torch, os, sys
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE = "cardiffnlp/twitter-roberta-base-sentiment"

ADAPTERS = {
    "twitter": {
        "path": "/tmp/twitter_v39_out/sentiment_analyzer/artifacts/models/twitter",
        "labels": {0: "negative", 1: "positive"},
        "num_labels": 2,
    },
    "amazon": {
        "path": "/tmp/amazon_v34_out/sentiment_analyzer/artifacts/models/amazon",
        "labels": {0: "negative", 1: "positive"},
        "num_labels": 2,
    },
}

def load_model(name):
    cfg = ADAPTERS[name]
    print(f"Loading {name} model...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=cfg["num_labels"], ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base, cfg["path"])
    model.eval()
    return tokenizer, model, cfg["labels"]

def predict(tokenizer, model, labels, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred = torch.argmax(probs).item()
    return labels[pred], probs[pred].item(), {labels[i]: probs[i].item() for i in range(len(labels))}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ADAPTERS:
        print(f"Usage: python {sys.argv[0]} <{'|'.join(ADAPTERS)}> [text|--interactive]")
        sys.exit(1)

    name = sys.argv[1]
    tokenizer, model, labels = load_model(name)

    if len(sys.argv) > 2 and sys.argv[2] == "--interactive":
        print(f"{name} model ready. Type text (Ctrl+Z to quit):")
        for line in sys.stdin:
            text = line.strip()
            if text:
                sentiment, conf, _ = predict(tokenizer, model, labels, text)
                print(f"  {sentiment} ({conf:.1%})")
    elif len(sys.argv) > 2:
        text = sys.argv[2]
        sentiment, conf, all_probs = predict(tokenizer, model, labels, text)
        probs_str = " | ".join(f"{k}: {v:.1%}" for k, v in all_probs.items())
        print(f"'{text}' -> {sentiment} ({conf:.1%})  [{probs_str}]")
    else:
        tests = [
            "This product is amazing! I love it!",
            "This is the worst thing I have ever bought.",
            "It is okay, nothing special.",
        ]
        for text in tests:
            sentiment, conf, _ = predict(tokenizer, model, labels, text)
            print(f"'{text[:45]:<45}' -> {sentiment:<8} ({conf:.1%})")
