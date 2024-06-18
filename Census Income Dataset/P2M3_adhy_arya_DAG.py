'''

=============================================================================
Milestone 3

Nama  : Adhy Arya Hendrata
Batch : FTDS-005-BSD

an Automate Script Program and Data Load from Postgresql into Elasticsearch, 
The Data i'm using about an Income Dataset based on United States. 
=============================================================================

'''




# Import Libraries
import logging
import psycopg2
import psycopg2 as db
import pandas as pd
import datetime as dt
import numpy as np
import warnings
warnings.filterwarnings("ignore")


from elasticsearch import Elasticsearch
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 5, 24, 15, 30),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'depends_on_past': False,
    'max_active_runs': 1,
    'catchup': False,
    'execution_timeout': timedelta(minutes=30)
}

# Function to fetch data from PostgreSQL
def fetch_data_from_postgresql():
    try:
        conn_string = "dbname='airflow' user='airflow' password='airflow' host='localhost' port='5432'"
        conn = psycopg2.connect(conn_string)
        df = pd.read_sql("SELECT * FROM table_m3", conn)
        df['id'] = range(1, len(df) + 1)
        df.to_csv('/opt/airflow/dags/P2M3_adhy_arya_data_raw.csv', index=False)
        conn.close()
        logging.info("Data fetched and saved to CSV successfully.")
    except Exception as e:
        logging.error("Error occurred while fetching data from PostgreSQL: %s", str(e))
        raise e

# Function for data cleaning
def cleaning_values_handling(df_cleaning):
    df_cleaning = df_cleaning.drop_duplicates()
    df_cleaning = df_cleaning.dropna()
    df_cleaning.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df_cleaning.columns = [col.replace('-', '_').lower() for col in df_cleaning.columns]
    df_cleaning = df_cleaning.rename(columns={'workclass': 'working_class', 'fnlwgt': 'final_weight'})
    df_cleaning.replace('-', ' ', regex=True, inplace=True)
    df_cleaning = df_cleaning.applymap(lambda x: x.replace('-', ' ').lower() if isinstance(x, str) else x)
    return df_cleaning

def cleaning_function(dataset):
    return cleaning_values_handling(dataset)

def using_cleaning_values_handling():
    try:
        json_file_path = '/opt/airflow/dags/json_data.json'
        df = pd.read_json(json_file_path)
        cleaned_df = cleaning_function(df)
        cleaned_df.to_csv('/opt/airflow/dags/P2M3_adhy_arya_data_clean.csv', index=False)
        cleaned_data_json = cleaned_df.to_json(orient='records')
        with open('/opt/airflow/dags/P2M3_adhy_arya_data_clean.json', 'w') as json_file:
            json_file.write(cleaned_data_json)
        logging.info("Cleaned data saved to /opt/airflow/dags/P2M3_adhy_arya_data_clean.csv")
        return cleaned_data_json
    except Exception as e:
        logging.error("Error occurred during data cleaning: %s", str(e))
        raise e

def transport_elasticsearch():
    try:
        es = Elasticsearch('http://elasticsearch:9200')
        df = pd.read_csv('/opt/airflow/dags/P2M3_adhy_arya_data_clean.csv')
        for i, r in df.iterrows():
            doc = r.to_json()
            res = es.index(index="p2m3_adhy_arya_data", doc_type="_doc", body=doc)
            logging.info("Data transported to Elasticsearch successfully: %s", res)
    except Exception as e:
        logging.error("Error occurred while transporting data to Elasticsearch: %s", str(e))
        raise e

# Define the DAG
with DAG(
    'project_m3',
    default_args=default_args,
    schedule_interval='3 6 0 * * *',
    catchup=False,
) as dag:

    getData = PythonOperator(
        task_id='fetch_data_postgresql',
        python_callable=fetch_data_from_postgresql
    )

    cleanData = PythonOperator(
        task_id='cleaning_data',
        python_callable=using_cleaning_values_handling
    )

    transferData = PythonOperator(
        task_id='transfering_data',
        python_callable=transport_elasticsearch
    )

# Set task dependencies
getData >> cleanData >> transferData