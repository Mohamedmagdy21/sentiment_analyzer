
import pandas as pd
import torch
import mlflow

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from evaluation.evaluators.base_evaluator import BaseEvaluator


class HuggingFaceEvaluator(BaseEvaluator):

    def __init__(
        self,
        model_dir: str
    ):

        self.model_dir = model_dir

        self.model = None
        self.tokenizer = None

    def load_model(self):

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_dir
            )
        )

        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.model_dir
            )
        )

        self.model.eval()

    def load_test_data(
        self,
        dataset_cfg
    ):

        test_df = pd.read_csv(
            dataset_cfg.processed_test_path
        )

        return test_df

    def predict(
        self,
        texts
    ):

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = self.model(
                **encodings
            )

        predictions = (
            outputs.logits
            .argmax(dim=-1)
            .cpu()
            .numpy()
        )

        return predictions

    def compute_metrics(
        self,
        labels,
        predictions
    ):

        accuracy = accuracy_score(
            labels,
            predictions
        )

        precision = precision_score(
            labels,
            predictions,
            average="weighted"
        )

        recall = recall_score(
            labels,
            predictions,
            average="weighted"
        )

        f1 = f1_score(
            labels,
            predictions,
            average="weighted"
        )

        return {
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        }

    def evaluate(
        self,
        dataset_cfg
    ):

        print(
            "Loading trained model..."
        )

        self.load_model()

        print(
            "Loading test dataset..."
        )

        test_df = self.load_test_data(
            dataset_cfg
        )

        predictions = self.predict(
            test_df["text"].tolist()
        )

        metrics = self.compute_metrics(
            test_df["label"].values,
            predictions
        )

        print(
            "Evaluation Results"
        )

        for metric, value in metrics.items():

            print(
                f"{metric}: {value:.4f}"
            )

        with mlflow.start_run(
            run_name="evaluation"
        ):

            mlflow.log_metrics(
                metrics
            )

        return metrics

