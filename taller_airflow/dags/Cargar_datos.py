from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.mysql.operators.mysql import MySqlOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime, timedelta
import pandas as pd
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG(
    '2-Cargar_data',
    default_args=default_args,
    description='DAG para cargar datos de penguins a MySQL sin preprocesamiento',
    schedule_interval=None, # Solo ejecución manual si es "None"
    start_date=datetime(2025, 3, 8),
    catchup=False
)


csv_file_path = '/opt/airflow/data/penguins_size.csv' 
database_name = 'airflow_db'
table_name = 'penguins'

def check_file_exists(**kwargs):
    if not os.path.isfile(csv_file_path):
        raise FileNotFoundError(f"El archivo CSV no existe en la ruta: {csv_file_path}")
    print(f"Archivo CSV encontrado: {csv_file_path}")
    return True

create_table_sql = f"""
USE {database_name};
CREATE TABLE IF NOT EXISTS {table_name} (
    id INT AUTO_INCREMENT PRIMARY KEY,
    species VARCHAR(50),
    island VARCHAR(50),
    culmen_length_mm FLOAT,
    culmen_depth_mm FLOAT,
    flipper_length_mm INT,
    body_mass_g INT,
    sex VARCHAR(10),
    fecha_carga TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def load_penguins_data(**kwargs):
    print(f"Cargando datos de penguins desde {csv_file_path} a la tabla {table_name}...")
    
    df = pd.read_csv(csv_file_path)
    print(f"CSV cargado en dataframe, filas: {len(df)}")
    mysql_hook = MySqlHook(mysql_conn_id='mysql_default')
    
    engine = mysql_hook.get_sqlalchemy_engine()
    df.to_sql(
        name=table_name,
        con=engine,
        schema=database_name,
        if_exists='append',
        index=False,
        chunksize=1000 #carga en lotes de 1000
    )
    
    count_query = f"SELECT COUNT(*) FROM {database_name}.{table_name}"
    records_count = mysql_hook.get_records(count_query)[0][0]
    
    print(f"Carga completada. Total de registros en la tabla: {records_count}")
    return records_count

# Definir las tareas en el DAG
check_csv_task = PythonOperator(
    task_id='check_csv_exists',
    python_callable=check_file_exists,
    dag=dag
)

create_table_task = MySqlOperator(
    task_id='create_penguins_table',
    mysql_conn_id='mysql_default',
    sql=create_table_sql,
    dag=dag
)

load_data_task = PythonOperator(
    task_id='load_penguins_data',
    python_callable=load_penguins_data,
    dag=dag
)

check_csv_task >> create_table_task >> load_data_task