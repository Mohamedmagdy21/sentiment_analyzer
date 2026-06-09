import os
import subprocess
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


def _hydra_preprocess(dataset_name: str):
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

    if result.returncode != 0:
        raise RuntimeError(
            f"Preprocessing failed for {dataset_name}\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )


def _push_and_wait(kernel_dir: str, **context):
    sys.path.insert(0, PROJECT_ROOT)

    from backend.kaggle_client import KaggleClient  # noqa: PLC0415

    client = KaggleClient()
    client.push_kernel(kernel_dir)
    client.wait_for_completion(interval=60)

    context["ti"].xcom_push(
        key="kernel_id",
        value=client._kernel_id,
    )


def _download_artifacts(dataset_name: str, **context):
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

    end = EmptyOperator(task_id="end")

    start >> [preprocess_twitter, preprocess_amazon]

    preprocess_twitter >> train_twitter_kaggle >> download_twitter_artifacts
    preprocess_amazon >> train_amazon_kaggle >> download_amazon_artifacts

    [download_twitter_artifacts, download_amazon_artifacts] >> end
