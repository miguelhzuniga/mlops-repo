from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

# Argumentos por defecto para el DAG
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2023, 1, 1),
}

# Definición del DAG
dag = DAG(
    'verificacion_configuracion',
    default_args=default_args,
    description='DAG para verificar la configuración correcta de Airflow',
    schedule_interval=None,
    catchup=False,
)

# Tarea 1: Verificar las variables de entorno
def verificar_variables():
    variables_necesarias = [
        'MLFLOW_TRACKING_URI', 
        'POSTGRES_RAW_DATA_CONN', 
        'POSTGRES_CLEAN_DATA_CONN',
        'MINIO_ENDPOINT',
        'MINIO_ACCESS_KEY',
        'MINIO_SECRET_KEY'
    ]
    
    resultado = {}
    for var in variables_necesarias:
        var_airflow = f'AIRFLOW_VAR__{var}'
        resultado[var] = os.environ.get(var_airflow, "No definida")
    
    print("Variables de entorno encontradas:")
    for var, valor in resultado.items():
        print(f"- {var}: {valor}")
    
    return resultado

tarea_verificar_variables = PythonOperator(
    task_id='verificar_variables',
    python_callable=verificar_variables,
    dag=dag,
)

# Tarea 2: Verificar directorios de DAGs
tarea_verificar_directorio = BashOperator(
    task_id='verificar_directorio',
    bash_command='echo "Contenido del directorio de DAGs:" && ls -la $AIRFLOW__CORE__DAGS_FOLDER',
    dag=dag,
)

# Tarea 3: Imprimir información de la configuración
tarea_info_configuracion = BashOperator(
    task_id='info_configuracion',
    bash_command='''
    echo "Información de configuración de Airflow:"
    echo "- Executor: $AIRFLOW__CORE__EXECUTOR"
    echo "- DAGs folder: $AIRFLOW__CORE__DAGS_FOLDER"
    echo "- Database connection: $AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"
    ''',
    dag=dag,
)

# Definir el orden de ejecución
tarea_verificar_variables >> tarea_verificar_directorio >> tarea_info_configuracion