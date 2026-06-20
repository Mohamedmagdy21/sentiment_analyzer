
import os
import pandas as pd
import mlflow

import transformers.utils.import_utils as _utils
_utils.check_torch_load_is_safe = lambda: None

from peft import PeftModel
from datetime import datetime

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
        model_dir: str,
        labels_map: dict
    ):

        self.model_dir = model_dir
        self.labels_map = labels_map

        self.model = None
        self.tokenizer = None

    def load_model(self):

        base_model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment",
         num_labels=3,
         ignore_mismatched_sizes=True
        )
        self.model = PeftModel.from_pretrained(base_model, self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        from inference.model_loader import device
        self.model.to(device)
        self.model.eval()

    def predict(self, texts, batch_size=64):
        from inference.model_loader import predict as shared_predict
        preds, _ = shared_predict(texts, self.tokenizer, self.model, batch_size=batch_size)
        return preds

    def load_test_data(
        self,
        dataset_cfg
    ):

        test_df = pd.read_csv(
            dataset_cfg.processed_test_path
        )

        return test_df

    def compute_metrics(
        self,
        labels,
        predictions
    ):

        accuracy = accuracy_score(
            labels,
            predictions
        )



        f1 = f1_score(
            labels,
            predictions,
            average="weighted"
        )

        return {
            "test_accuracy": accuracy,
            "test_f1": f1
        }


    def compute_recall_precision(
        self,
        labels,
        predictions
    ):

        precision = precision_score(
            labels,
            predictions,
            average=None,
            labels=[0,1,2],
            zero_division=0
        )

        recall = recall_score(
            labels,
            predictions,
            average=None,
            labels=[0,1,2],
            zero_division=0
        )

        return {"precision": precision, "recall": recall}


    def evaluate(
        self,
        dataset_cfg
    ):

        os.system("pkill -f 'uvicorn inference.main' > /dev/null 2>&1")

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

        recall_precision = self.compute_recall_precision(
            test_df["label"].values,
            predictions
        )

        results_file = os.path.join(
        self.model_dir,
        f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        with open(results_file, "w") as f:
            f.write("Overall Metrics\n")
            f.write("-----------------\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write("\nPer-Class Metrics\n")
            f.write("-----------------\n")
            for class_id, class_name in self.labels_map.items():
                f.write(
                    f"precision_{class_name}: "
                    f"{recall_precision['precision'][class_id]:.4f}\n"
                )
                f.write(
                    f"recall_{class_name}: "
                    f"{recall_precision['recall'][class_id]:.4f}\n"
                )

        try:
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
            with mlflow.start_run(run_name="evaluation"):
                mlflow.log_metrics(metrics)
        except Exception as e:
            print(f"MLflow logging skipped: {e}")

        return metrics

