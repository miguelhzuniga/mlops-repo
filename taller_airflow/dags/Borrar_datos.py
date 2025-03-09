from airflow import DAG
from airflow.providers.mysql.operators.mysql import MySqlOperator
from datetime import datetime, timedelta


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}


dag = DAG(
    '1-Borrar_penguins_db',
    default_args=default_args,
    description='DAG para borrar contenido de la base de datos de penguins',
    schedule_interval=None,  # Solo ejecución manual si es "None"
    start_date=datetime(2025, 3, 8),
    catchup=False
)

database_name = 'airflow_db'
table_name = 'penguins'

check_and_drop_table = f"""
USE {database_name};
DROP TABLE IF EXISTS {table_name};
"""


drop_table_task = MySqlOperator(
    task_id='drop_penguins_table',
    mysql_conn_id='mysql_default',
    sql=check_and_drop_table,
    dag=dag
)

log_completion = MySqlOperator(
    task_id='log_completion',
    mysql_conn_id='mysql_default',
    sql=f"SELECT 'Base de datos de penguins borrada exitosamente' as log_message",
    dag=dag
)

drop_table_task >> log_completion