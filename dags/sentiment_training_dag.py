import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


def _log_duration(name: str, start: float):
    elapsed = time.perf_counter() - start
    print(f"[TIMING] {name} completed in {elapsed:.2f}s")


def _hydra_preprocess(dataset_name: str):
    start = time.perf_counter()
    print(f"[TIMING] Preprocessing {dataset_name} started")

    cmd = [
        sys.executable,
        "-m",
        "preprocessing.preprocess",
        f"dataset={dataset_name}",
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


def _push_and_wait(kernel_dir: str, **context):
    start = time.perf_counter()
    dataset_name = kernel_dir.rstrip("/").split("/")[-1]
    print(f"[TIMING] Kaggle training for {dataset_name} started")

    sys.path.insert(0, PROJECT_ROOT)

    from backend.kaggle_client import KaggleClient  # noqa: PLC0415

    client = KaggleClient()
    client.push_kernel(kernel_dir)
    client.wait_for_completion(interval=60)

    context["ti"].xcom_push(
        key="kernel_id",
        value=client._kernel_id,
    )

    _log_duration(f"kaggle training {dataset_name}", start)


def _download_artifacts(dataset_name: str, **context):
    start = time.perf_counter()
    print(f"[TIMING] Download artifacts for {dataset_name} started")

    sys.path.insert(0, PROJECT_ROOT)

    from backend.kaggle_client import KaggleClient  # noqa: PLC0415

    kernel_ref = context["ti"].xcom_pull(
        task_ids=f"train_{dataset_name}_kaggle",
        key="kernel_id",
    )

    client = KaggleClient()
    client._kernel_id = kernel_ref

    artifact_dir = f"{PROJECT_ROOT}/artifacts/models/{dataset_name}"
    log_dir = f"{PROJECT_ROOT}/artifacts/logs/{dataset_name}"

    client.download_output(kernel_ref, artifact_dir)
    client.download_log(kernel_ref, log_dir)

    _log_duration(f"download artifacts {dataset_name}", start)


def _hydra_evaluate(dataset_name: str):
    start = time.perf_counter()
    print(f"[TIMING] Evaluation for {dataset_name} started")

    model_dir = f"{PROJECT_ROOT}/artifacts/models/{dataset_name}"
    model_cfg = f"model={dataset_name}_roberta"

    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        f"dataset={dataset_name}",
        model_cfg,
        f"evaluator.model_dir={model_dir}",
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
        "in parallel: preprocess, train on Kaggle GPU, download artifacts."
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

    train_twitter_kaggle = PythonOperator(
        task_id="train_twitter_kaggle",
        python_callable=_push_and_wait,
        op_kwargs={"kernel_dir": f"{PROJECT_ROOT}/kaggle/twitter_training"},
    )

    train_amazon_kaggle = PythonOperator(
        task_id="train_amazon_kaggle",
        python_callable=_push_and_wait,
        op_kwargs={"kernel_dir": f"{PROJECT_ROOT}/kaggle/amazon_training"},
    )

    download_twitter_artifacts = PythonOperator(
        task_id="download_twitter_artifacts",
        python_callable=_download_artifacts,
        op_kwargs={"dataset_name": "twitter"},
    )

    download_amazon_artifacts = PythonOperator(
        task_id="download_amazon_artifacts",
        python_callable=_download_artifacts,
        op_kwargs={"dataset_name": "amazon"},
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

    start >> [preprocess_twitter, preprocess_amazon]

    preprocess_twitter >> train_twitter_kaggle >> download_twitter_artifacts >> evaluate_twitter
    preprocess_amazon >> train_amazon_kaggle >> download_amazon_artifacts >> evaluate_amazon

    [evaluate_twitter, evaluate_amazon] >> end
