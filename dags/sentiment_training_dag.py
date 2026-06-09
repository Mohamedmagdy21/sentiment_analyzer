from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime


with DAG(
    dag_id="sentiment_training",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False
) as dag:

    preprocess = BashOperator(
        task_id="preprocess_data",
        bash_command="""
        python preprocessing/preprocess_twitter.py
        """
    )

    trigger_kaggle_training = PythonOperator(
    task_id="trigger_kaggle_training",
    python_callable=trigger_kaggle_training
)

    train = BashOperator(
        task_id="train_model",
        bash_command="""
        python training/train.py
        """
    )

    evaluate = BashOperator(
        task_id="evaluate_model",
        bash_command="""
        python evaluation/evaluate.py
        """
    )

    preprocess >> train >> evaluate