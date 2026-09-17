from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def check_new_data():
    print("Checking for new fMRI data...")
    return True

def load_to_dw():
    print("Loading processed features into Data Warehouse...")
    return True

with DAG(
    'fmri_etl_pipeline',
    default_args=default_args,
    description='ETL pipeline for fMRI Age Prediction',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    catchup=False,
) as dag:

    detect_data = PythonOperator(
        task_id='detect_new_data',
        python_callable=check_new_data,
    )

    trigger_spark = BashOperator(
        task_id='trigger_spark_job',
        bash_command='spark-submit /opt/airflow/src/spark_transform.py --input /data/raw --output /data/processed',
    )

    load_dw = PythonOperator(
        task_id='load_to_dw',
        python_callable=load_to_dw,
    )

    detect_data >> trigger_spark >> load_dw
