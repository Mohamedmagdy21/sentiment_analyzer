import os
import subprocess
import time
from datetime import datetime, timedelta

TRAIN_TIMEOUT = timedelta(hours=2)

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)
PROJECT_PYTHON = os.path.join(PROJECT_ROOT, "venv_cuda", "bin", "python3")


def _log_duration(name: str, start: float):
    elapsed = time.perf_counter() - start
    print(f"[TIMING] {name} completed in {elapsed:.2f}s")


def _stream_subprocess(cmd, cwd):
    """Run subprocess streaming output live to Airflow logs."""
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


def _hydra_preprocess(dataset_name: str):
    start = time.perf_counter()
    print(f"[TIMING] Preprocessing {dataset_name} started")

    cmd = [
        PROJECT_PYTHON,
        "-m",
        "preprocessing.preprocess",
        f"dataset={dataset_name}",
        f"hydra.run.dir=/tmp/hydra_preprocess_{dataset_name}",
    ]
    if dataset_name == "amazon":
        cmd += [
            "preprocessing=amazon_preprocessor",
            "model=amazon_roberta",
        ]

    rc = _stream_subprocess(cmd, PROJECT_ROOT)

    if rc != 0:
        raise RuntimeError(f"Preprocessing failed for {dataset_name} (rc={rc})")

    _log_duration(f"preprocessing {dataset_name}", start)


def _hydra_train(dataset_name: str):
    import shutil

    # Clear Hydra's saved state from previous runs
    for hydra_dir in [
        os.path.join(PROJECT_ROOT, ".hydra"),
        f"/tmp/hydra_train_{dataset_name}",
    ]:
        if os.path.exists(hydra_dir):
            shutil.rmtree(hydra_dir)
            print(f"[INFO] Cleared {hydra_dir}")

    # Clear checkpoints to prevent HuggingFace Trainer from resuming
    checkpoint_dir = os.path.join(PROJECT_ROOT, "artifacts", "checkpoints")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print(f"[INFO] Cleared checkpoints dir")

    start = time.perf_counter()
    print(f"[TIMING] Training {dataset_name} started")

    model_cfg = f"model={dataset_name}_roberta"
    cmd = [
        PROJECT_PYTHON,
        "-m",
        "training.train",
        f"dataset={dataset_name}",
        model_cfg,
        f"hydra.run.dir=/tmp/hydra_train_{dataset_name}",
        "--config-dir", os.path.join(PROJECT_ROOT, "configs"),
        "--config-name", "config",
    ]

    rc = _stream_subprocess(cmd, PROJECT_ROOT)

    if rc != 0:
        raise RuntimeError(f"Training failed for {dataset_name} (rc={rc})")

    _log_duration(f"training {dataset_name}", start)

    

def _hydra_generate_semantic_baseline(dataset_name: str):
    import yaml

    start = time.perf_counter()
    print(f"[TIMING] Semantic baseline for {dataset_name} started")

    config_dir = os.path.join(PROJECT_ROOT, "configs")

    with open(os.path.join(config_dir, "dataset", f"{dataset_name}.yaml")) as f:
        ds_cfg = yaml.safe_load(f)

    train_csv = os.path.join(PROJECT_ROOT, ds_cfg["processed_train_path"])
    text_col = "text"
    label_col = "label"

    import pandas as pd
    df = pd.read_csv(train_csv)
    texts = df[text_col].dropna().astype(str).tolist()
    labels = df[label_col].dropna().values if label_col in df.columns else None

    from inference.semantic_monitoring_utils import fit_semantic_baseline
    fit_semantic_baseline(dataset_name, texts, labels=labels)

    _log_duration(f"semantic_baseline {dataset_name}", start)


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
        f"hydra.run.dir=/tmp/hydra_evaluate_{dataset_name}",
    ]

    rc = _stream_subprocess(cmd, PROJECT_ROOT)

    if rc != 0:
        raise RuntimeError(f"Evaluation failed for {dataset_name} (rc={rc})")

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

    # Gate: ensures train_twitter fires only once
    # after BOTH preprocessing tasks complete
    preprocessing_done = EmptyOperator(task_id="preprocessing_done")

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

    generate_baseline_twitter = PythonOperator(
        task_id="generate_baseline_twitter",
        python_callable=_hydra_generate_semantic_baseline,
        op_kwargs={"dataset_name": "twitter"},
    )

    generate_baseline_amazon = PythonOperator(
        task_id="generate_baseline_amazon",
        python_callable=_hydra_generate_semantic_baseline,
        op_kwargs={"dataset_name": "amazon"},
    )

    end = EmptyOperator(task_id="end")

   # Twitter pipeline first
    start >> preprocess_twitter
    preprocess_twitter >> train_twitter
    train_twitter >> evaluate_twitter
    evaluate_twitter >> generate_baseline_twitter

    # Amazon pipeline after twitter completes
    generate_baseline_twitter >> preprocess_amazon
    preprocess_amazon >> train_amazon
    train_amazon >> evaluate_amazon
    evaluate_amazon >> generate_baseline_amazon

    end << [generate_baseline_amazon]