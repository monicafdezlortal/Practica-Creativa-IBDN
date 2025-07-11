import sys, os, re

from airflow import DAG
from airflow.operators.bash import BashOperator

from datetime import datetime, timedelta
import iso8601

PROJECT_HOME = "/home/monica.fernandez/practica_creativa"


default_args = {
  'owner': 'airflow',
  'depends_on_past': False,
  'start_date': iso8601.parse_date("2016-12-01"),
  'retries': 3,
  'retry_delay': timedelta(minutes=5),
}

training_dag = DAG(
  'agile_data_science_batch_prediction_model_training',
  default_args=default_args,
  schedule_interval=None
)

# Bash command templates
pyspark_bash_command = """
source /home/monica.fernandez/practica_creativa/venv-airflow/bin/activate && \
export PYSPARK_PYTHON=/home/monica.fernandez/practica_creativa/venv-airflow/bin/python && \
spark-submit \
  --master {{ params.master }} \
  --packages com.datastax.spark:spark-cassandra-connector_2.12:3.5.0,com.github.jnr:jnr-posix:3.1.15 \
  {{ params.base_path }}/{{ params.filename }} \
  {{ params.base_path }}
"""



# Train the classifier model
train_classifier_model_operator = BashOperator(
  task_id = "pyspark_train_classifier_model",
  bash_command = pyspark_bash_command,
  params = {
    "master": "local[4]",
    "filename": "resources/train_spark_mllib_model.py",
    "base_path": "/home/monica.fernandez/practica_creativa"
  
  },
  



  execution_timeout=timedelta(minutes=30), 
  dag=training_dag
)

dag = training_dag
