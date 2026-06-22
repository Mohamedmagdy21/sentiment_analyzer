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
    process = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()
    return process.returncode


def _hydra_train(dataset_name: str):
    """Run PEFT LoRA training via Hydra for the given dataset."""
    import shutil
    for hydra_dir in [
        os.path.join(PROJECT_ROOT, ".hydra"),
        f"/tmp/hydra_train_{dataset_name}",
    ]:
        if os.path.exists(hydra_dir):
            shutil.rmtree(hydra_dir)
    checkpoint_dir = os.path.join(PROJECT_ROOT, "artifacts", "checkpoints")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    start = time.perf_counter()
    cmd = [
        PROJECT_PYTHON, "-m", "training.train",
        f"dataset={dataset_name}",
        f"model={dataset_name}_roberta",
        f"hydra.run.dir=/tmp/hydra_train_{dataset_name}",
        "--config-dir", os.path.join(PROJECT_ROOT, "configs"),
        "--config-name", "config",
    ]
    rc = _stream_subprocess(cmd, PROJECT_ROOT)
    if rc != 0:
        raise RuntimeError(f"Training failed for {dataset_name} (rc={rc})")
    _log_duration(f"training {dataset_name}", start)


def _hydra_evaluate(dataset_name: str):
    """Evaluate the fine-tuned PEFT model on the test set."""
    start = time.perf_counter()
    model_dir = f"{PROJECT_ROOT}/artifacts/models/{dataset_name}"
    cmd = [
        PROJECT_PYTHON, "-m", "evaluation.evaluate",
        f"dataset={dataset_name}",
        f"model={dataset_name}_roberta",
        f"evaluator.model_dir={model_dir}",
        "evaluator.use_peft=True",
        f"hydra.run.dir=/tmp/hydra_evaluate_{dataset_name}",
    ]
    rc = _stream_subprocess(cmd, PROJECT_ROOT)
    if rc != 0:
        raise RuntimeError(f"Evaluation failed for {dataset_name} (rc={rc})")
    _log_duration(f"evaluation {dataset_name}", start)


def _generate_drift_baselines(dataset_name: str):
    """Compute and save data/target/prediction drift baselines from train+val sets."""
    start = time.perf_counter()
    import pandas as pd
    import numpy as np
    import torch
    from inference.model_loader import load_model, predict
    from inference.monitoring_utils import generate_and_save_baselines

    train_df = pd.read_csv(os.path.join(PROJECT_ROOT, f"Data/processed/{dataset_name}/train.csv"))
    val_df = pd.read_csv(os.path.join(PROJECT_ROOT, f"Data/processed/{dataset_name}/val.csv"))

    tokenizer, model = load_model(dataset_name)
    model.eval()
    val_texts = val_df["text"].dropna().astype(str).tolist()
    _, probs = predict(val_texts, tokenizer, model, batch_size=16)
    confidences = probs.max(axis=1)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    generate_and_save_baselines(
        dataset_name, train_df, val_df,
        target_col="label", val_confidences=confidences,
    )
    _log_duration(f"drift_baselines {dataset_name}", start)


def _deploy():
    """Copy trained adapter weights from artifacts to the inference directory."""
    start = time.perf_counter()
    import shutil
    src = os.path.join(PROJECT_ROOT, "artifacts", "models")
    dst = os.path.join(PROJECT_ROOT, "inference", "artifacts", "models")
    os.makedirs(dst, exist_ok=True)
    for model in ["twitter", "amazon"]:
        src_m = os.path.join(src, model)
        dst_m = os.path.join(dst, model)
        if os.path.exists(src_m):
            if os.path.exists(dst_m):
                shutil.rmtree(dst_m)
            shutil.copytree(src_m, dst_m)
    print("[DEPLOY] Adapters copied to inference directory.")
    _log_duration("deploy", start)


# Manual trigger only — trains PEFT adapters, evaluates, generates drift baselines, deploys
with DAG(
    dag_id="peft_training",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(seconds=120)},
    description="PEFT LoRA training + drift baselines + deploy",
) as dag:

    start = EmptyOperator(task_id="start")

    train_twitter = PythonOperator(
        task_id="train_twitter",
        python_callable=_hydra_train,
        op_kwargs={"dataset_name": "twitter"},
        execution_timeout=TRAIN_TIMEOUT,
    )

    evaluate_twitter = PythonOperator(
        task_id="evaluate_twitter",
        python_callable=_hydra_evaluate,
        op_kwargs={"dataset_name": "twitter"},
    )

    drift_twitter = PythonOperator(
        task_id="generate_drift_baselines_twitter",
        python_callable=_generate_drift_baselines,
        op_kwargs={"dataset_name": "twitter"},
    )

    train_amazon = PythonOperator(
        task_id="train_amazon",
        python_callable=_hydra_train,
        op_kwargs={"dataset_name": "amazon"},
        execution_timeout=TRAIN_TIMEOUT,
    )

    evaluate_amazon = PythonOperator(
        task_id="evaluate_amazon",
        python_callable=_hydra_evaluate,
        op_kwargs={"dataset_name": "amazon"},
    )

    drift_amazon = PythonOperator(
        task_id="generate_drift_baselines_amazon",
        python_callable=_generate_drift_baselines,
        op_kwargs={"dataset_name": "amazon"},
    )

    deploy = PythonOperator(
        task_id="deploy_adapters",
        python_callable=_deploy,
    )

    end = EmptyOperator(task_id="end")

    start >> train_twitter >> evaluate_twitter >> drift_twitter
    drift_twitter >> train_amazon >> evaluate_amazon >> drift_amazon
    drift_amazon >> deploy >> end
