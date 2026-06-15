import os
import subprocess
import time
from datetime import datetime, timedelta

TRAIN_TIMEOUT = timedelta(hours=2)

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)
PROJECT_PYTHON = os.path.join(PROJECT_ROOT, "venv_cuda", "bin", "python3")


def _log_duration(name: str, start: float):
    elapsed = time.perf_counter() - start
    print(f"[TIMING] {name} completed in {elapsed:.2f}s")


def _hydra_preprocess(dataset_name: str):
    start = time.perf_counter()
    print(f"[TIMING] Preprocessing {dataset_name} started")

    cmd = [
        PROJECT_PYTHON,
        "-m",
        "preprocessing.preprocess",
        f"dataset={dataset_name}",
        "hydra.run.dir=.",
    ]
    if dataset_name == "amazon":
        cmd += [
            "preprocessing=amazon_preprocessor",
            "model=amazon_roberta",
        ]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Preprocessing failed for {dataset_name}\n"
            f"STDERR: {result.stderr}"
        )

    _log_duration(f"preprocessing {dataset_name}", start)


def _hydra_train(dataset_name: str):
    start = time.perf_counter()
    print(f"[TIMING] Training {dataset_name} started")

    model_cfg = f"model={dataset_name}_roberta"
    cmd = [
        PROJECT_PYTHON,
        "-m",
        "training.train",
        f"dataset={dataset_name}",
        model_cfg,
        "hydra.run.dir=.",
    ]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Training failed for {dataset_name}\n"
            f"STDERR: {result.stderr}"
        )

    _log_duration(f"training {dataset_name}", start)


def _hydra_evaluate(dataset_name: str):
    start = time.perf_counter()
    print(f"[TIMING] Evaluation for {dataset_name} started")

    model_dir = f"{PROJECT_ROOT}/artifacts/models/{dataset_name}"
    model_cfg = f"model={dataset_name}_roberta"

    cmd = [
        PROJECT_PYTHON,
        "-m",
        "evaluation.evaluate",
        f"dataset={dataset_name}",
        model_cfg,
        f"evaluator.model_dir={model_dir}",
        "hydra.run.dir=.",
    ]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Evaluation failed for {dataset_name}\n"
            f"STDERR: {result.stderr}"
        )

    _log_duration(f"evaluation {dataset_name}", start)


with DAG(
    dag_id="sentiment_training",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(seconds=120),
    },
    description=(
        "Orchestrate Twitter and Amazon sentiment model pipelines "
        "in parallel: preprocess, train locally on GPU, evaluate."
    ),
) as dag:

    start = EmptyOperator(task_id="start")

    preprocess_twitter = PythonOperator(
        task_id="preprocess_twitter",
        python_callable=_hydra_preprocess,
        op_kwargs={"dataset_name": "twitter"},
    )

    preprocess_amazon = PythonOperator(
        task_id="preprocess_amazon",
        python_callable=_hydra_preprocess,
        op_kwargs={"dataset_name": "amazon"},
    )

    train_twitter = PythonOperator(
        task_id="train_twitter",
        python_callable=_hydra_train,
        op_kwargs={"dataset_name": "twitter"},
        execution_timeout=TRAIN_TIMEOUT,
    )

    train_amazon = PythonOperator(
        task_id="train_amazon",
        python_callable=_hydra_train,
        op_kwargs={"dataset_name": "amazon"},
        execution_timeout=TRAIN_TIMEOUT,
    )

    evaluate_twitter = PythonOperator(
        task_id="evaluate_twitter",
        python_callable=_hydra_evaluate,
        op_kwargs={"dataset_name": "twitter"},
    )

    evaluate_amazon = PythonOperator(
        task_id="evaluate_amazon",
        python_callable=_hydra_evaluate,
        op_kwargs={"dataset_name": "amazon"},
    )

    end = EmptyOperator(task_id="end")

    # Preprocessing runs in parallel (CPU-only)
    start >> [preprocess_twitter, preprocess_amazon]

    # Training is serialized to avoid GPU OOM (4GB VRAM)
    preprocess_twitter >> train_twitter
    preprocess_amazon >> train_twitter
    train_twitter >> train_amazon

    # Evaluation each right after its training finishes
    train_twitter >> evaluate_twitter
    train_amazon >> evaluate_amazon

    end << [evaluate_twitter, evaluate_amazon]
