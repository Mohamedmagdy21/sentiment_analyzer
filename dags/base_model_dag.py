import os
import subprocess
import time
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)
PROJECT_PYTHON = os.path.join(PROJECT_ROOT, "venv_cuda", "bin", "python3")


def _log_duration(name: str, start: float):
    """Log elapsed time for a pipeline step."""
    elapsed = time.perf_counter() - start
    print(f"[TIMING] {name} completed in {elapsed:.2f}s")


def _stream_subprocess(cmd, cwd):
    """Run a subprocess and stream stdout line-by-line."""
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()
    return process.returncode


def _preprocess(dataset_name: str):
    """Run preprocessing pipeline for the given dataset."""
    start = time.perf_counter()
    cmd = [
        PROJECT_PYTHON, "-m", "preprocessing.preprocess",
        f"dataset={dataset_name}",
        f"hydra.run.dir=/tmp/hydra_preprocess_{dataset_name}",
    ]
    if dataset_name == "amazon":
        cmd += ["preprocessing=amazon_preprocessor", "model=amazon_roberta"]
    rc = _stream_subprocess(cmd, PROJECT_ROOT)
    if rc != 0:
        raise RuntimeError(f"Preprocessing failed for {dataset_name} (rc={rc})")
    _log_duration(f"preprocessing {dataset_name}", start)


def _evaluate_base(dataset_name: str):
    """Evaluate the base (unfine-tuned) model on the given dataset."""
    start = time.perf_counter()
    model_dir = f"{PROJECT_ROOT}/artifacts/models/{dataset_name}"
    cmd = [
        PROJECT_PYTHON, "-m", "evaluation.evaluate",
        f"dataset={dataset_name}",
        f"model={dataset_name}_roberta",
        f"evaluator.model_dir={model_dir}",
        "evaluator.use_peft=False",
        f"hydra.run.dir=/tmp/hydra_evaluate_base_{dataset_name}",
    ]
    rc = _stream_subprocess(cmd, PROJECT_ROOT)
    if rc != 0:
        raise RuntimeError(f"Base evaluation failed for {dataset_name} (rc={rc})")
    _log_duration(f"evaluate_base {dataset_name}", start)


def _generate_semantic_baseline(dataset_name: str):
    """Fit and save a semantic drift baseline from training set embeddings."""
    start = time.perf_counter()
    cmd = [
        "env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        PROJECT_PYTHON, "-c",
        f"from inference.semantic_monitoring_utils import fit_semantic_baseline; "
        f"import pandas as pd; "
        f"df = pd.read_csv('{PROJECT_ROOT}/Data/processed/{dataset_name}/train.csv'); "
        f"fit_semantic_baseline('{dataset_name}', df['text'].dropna().astype(str).tolist(), "
        f"labels=df['label'].values if 'label' in df.columns else None)",
    ]
    rc = _stream_subprocess(cmd, PROJECT_ROOT)
    if rc != 0:
        raise RuntimeError(f"Semantic baseline failed for {dataset_name} (rc={rc})")
    _log_duration(f"semantic_baseline {dataset_name}", start)


# Pipeline: preprocess -> base eval -> semantic baselines, sequential twitter -> amazon
with DAG(
    dag_id="base_model_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=60)},
    description="Preprocess, evaluate base model, generate semantic baselines",
) as dag:

    start = EmptyOperator(task_id="start")

    preprocess_twitter = PythonOperator(
        task_id="preprocess_twitter",
        python_callable=_preprocess,
        op_kwargs={"dataset_name": "twitter"},
    )

    evaluate_base_twitter = PythonOperator(
        task_id="evaluate_base_twitter",
        python_callable=_evaluate_base,
        op_kwargs={"dataset_name": "twitter"},
    )

    semantic_twitter = PythonOperator(
        task_id="generate_semantic_baseline_twitter",
        python_callable=_generate_semantic_baseline,
        op_kwargs={"dataset_name": "twitter"},
    )

    preprocess_amazon = PythonOperator(
        task_id="preprocess_amazon",
        python_callable=_preprocess,
        op_kwargs={"dataset_name": "amazon"},
    )

    evaluate_base_amazon = PythonOperator(
        task_id="evaluate_base_amazon",
        python_callable=_evaluate_base,
        op_kwargs={"dataset_name": "amazon"},
    )

    semantic_amazon = PythonOperator(
        task_id="generate_semantic_baseline_amazon",
        python_callable=_generate_semantic_baseline,
        op_kwargs={"dataset_name": "amazon"},
    )

    end = EmptyOperator(task_id="end")

    start >> preprocess_twitter >> evaluate_base_twitter >> semantic_twitter
    semantic_twitter >> preprocess_amazon >> evaluate_base_amazon >> semantic_amazon
    semantic_amazon >> end
