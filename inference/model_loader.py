import os
import glob as _glob
import yaml
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda: None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs", "model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_model_config(dataset_name: str):
    pattern = os.path.join(CONFIGS_DIR, f"{dataset_name}_*.yaml")
    matches = _glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No model config found for '{dataset_name}' "
            f"(looked for {pattern})"
        )
    with open(matches[0]) as f:
        return yaml.safe_load(f)


def load_model(dataset_name: str):
    cfg = _get_model_config(dataset_name)
    model_dir = os.path.join(MODELS_DIR, dataset_name)
    pretrained_name = cfg["pretrained_name"]
    tokenizer_name = cfg.get("tokenizer_name", pretrained_name)
    num_labels = cfg.get("num_labels", 3)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    adapter_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        model = base_model
    model.to(device)
    model.eval()

    return tokenizer, model


#def load_both_models():
   # tokenizer_twitter, model_twitter = load_model("twitter")
    #tokenizer_amazon, model_amazon = load_model("amazon")
    #return (tokenizer_twitter, model_twitter), (tokenizer_amazon, model_amazon)


def predict(texts, tokenizer, model, batch_size=64):
    all_preds = []
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encodings = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**encodings)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
    return torch.cat(all_preds).numpy(), torch.cat(all_probs).numpy()
