from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.mysql.operators.mysql import MySqlOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime, timedelta
import pandas as pd
import os
import requests
import time
import json

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

def my_task(run_id):
    print(f"Ejecutando iteración {run_id}")

dag = DAG(
    'Cargar_data',
    default_args=default_args,
    description='DAG para cargar datos desde el servidor a MySQL sin preprocesamiento',
    schedule_interval=timedelta(seconds=5), # Solo ejecución manual si es "None"
    start_date=datetime(2025, 3, 26,0,0,0),
    catchup=False
)



# csv_file_path = '/opt/airflow/data/penguins_size.csv' 
database_name = 'airflow_db'
table_name = 'covertype'

# def check_file_exists(**kwargs):
#     if not os.path.isfile(csv_file_path):
#         raise FileNotFoundError(f"El archivo CSV no existe en la ruta: {csv_file_path}")
#     print(f"Archivo CSV encontrado: {csv_file_path}")
#     return True

# create_table_sql = f"""
# USE {database_name};
# CREATE TABLE IF NOT EXISTS {table_name} (
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     species VARCHAR(50),
#     island VARCHAR(50),
#     culmen_length_mm FLOAT,
#     culmen_depth_mm FLOAT,
#     flipper_length_mm INT,
#     body_mass_g INT,
#     sex VARCHAR(10),
#     fecha_carga TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );
# """

create_table_sql = f"""
USE {database_name};
CREATE TABLE IF NOT EXISTS {table_name} (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Elevation INT NOT NULL, 
    Aspect INT NOT NULL, 
    Slope INT NOT NULL, 
    Horizontal_Distance_To_Hydrology INT NOT NULL,
    Vertical_Distance_To_Hydrology INT NOT NULL,
    Horizontal_Distance_To_Roadways INT NOT NULL,
    Hillshade_9am INT NOT NULL,
    Hillshade_Noon INT NOT NULL,
    Hillshade_3pm INT NOT NULL,
    Horizontal_Distance_To_Fire_Points INT NOT NULL, 
    Wilderness_Area VARCHAR(50) NOT NULL,  
    Soil_Type VARCHAR(50) NOT NULL,  
    Cover_Type INT NOT NULL
);
"""

server_url = "http://api_server:80/data"
def server_response(server_url,group_number=1):
    
    # time.sleep(5)  # Espera a que el servidor se inicie

    try:
        #response = requests.get(server_url)
        params = {"group_number": group_number}

        response = requests.get(server_url, params=params)
        
        return response
        #print("Respuesta del servidor:", response.json())
    except Exception as e:
        print("Error:", e)


def load_data(**kwargs):
    print(f"Cargando datos desde servidor a la tabla {table_name}...")
    
    # df = pd.read_csv(csv_file_path)
    # print(f"CSV cargado en dataframe, filas: {len(df)}")
    data = json.loads(server_response)
    df = pd.DataFrame.from_dict(data)
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
# check_csv_task = PythonOperator(
#     task_id='check_csv_exists',
#     python_callable=check_file_exists,
#     dag=dag
# )

create_table_task = MySqlOperator(
    task_id='create_table',
    mysql_conn_id='mysql_default',
    sql=create_table_sql,
    dag=dag
)

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)

create_table_task >> load_data_task