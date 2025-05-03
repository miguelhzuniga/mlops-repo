import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.dummy import DummyOperator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Configuración de variables de entorno
HOST_IP = os.getenv('HOST_IP')
MINIO_ENDPOINT = os.getenv('MLFLOW_S3_ENDPOINT_URL')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI')
ARTIFACT_ROOT = f's3://mlflow/'

# Función para configurar tracking de MLflow
def set_mlflow_tracking(**kwargs):
    """Configurar tracking de MLflow"""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment('diabetes_experiment')
    
    # Configurar credenciales de almacenamiento
    os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_KEY
    
    print("Tracking de MLflow configurado exitosamente")

# Función para cargar datos desde PostgreSQL
def load_data(**kwargs):
    """Cargar datos desde PostgreSQL para entrenamiento de modelos"""
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    query = """
    SELECT * FROM clean_data.diabetes_train 
    WHERE batch_id = (
        SELECT MIN(batch_id) 
        FROM clean_data.batch_info
        WHERE batch_id IS NOT NULL
    )
    """
    
    df = pg_hook.get_pandas_df(query)
    
    if df.empty:
        print("No hay más datos para procesar")
        return {'continue_processing': False}
    
    columns_to_drop = ['id', 'batch_id', 'dataset']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    y = df['readmitted'].tolist()
    X = df.drop('readmitted', axis=1)
    
    X_dict = X.to_dict('records')
    
    kwargs['ti'].xcom_push(key='X_train', value=X_dict)
    kwargs['ti'].xcom_push(key='y_train', value=y)
    
    return {
        'X_train': X_dict,
        'y_train': y,
        'continue_processing': True
    }


def preprocess_data(**kwargs):
    """Preprocesar datos de entrenamiento"""
    X_train_dict = kwargs['ti'].xcom_pull(key='X_train')
    y_train = kwargs['ti'].xcom_pull(key='y_train')
    
    X_train = pd.DataFrame(X_train_dict)
    y_train = pd.Series(y_train)
    
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    X_train_processed = preprocessor.fit_transform(X_train)
    
    X_train_processed_list = X_train_processed.toarray().tolist()
    
    kwargs['ti'].xcom_push(key='X_train_processed', value=X_train_processed_list)
    kwargs['ti'].xcom_push(key='y_train', value=y_train.tolist())
    
    return {
        'X_train_processed': X_train_processed_list,
        'y_train': y_train.tolist()
    }

def train_models(**kwargs):
    """Entrenar múltiples modelos y registrar en MLflow"""
    X_train_processed_list = kwargs['ti'].xcom_pull(key='X_train_processed')
    y_train = kwargs['ti'].xcom_pull(key='y_train')
    
    X_train_processed = np.array(X_train_processed_list)
    y_train = np.array(y_train)
    
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
    }

    def evaluate_model(y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }

    set_mlflow_tracking()

    best_model = None
    best_score = 0
    best_model_obj = None
    model_metrics = {} 
    client = mlflow.tracking.MlflowClient()
    
    with mlflow.start_run() as main_run:
        run_id = main_run.info.run_id
        
        for name, model in models.items():
            print(f"Entrenando modelo: {name}")
            with mlflow.start_run(nested=True):
                model.fit(X_train_processed, y_train)
                y_pred = model.predict(X_train_processed)
                metrics = evaluate_model(y_train, y_pred)
                
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                mlflow.log_param('model_name', name)
                
                try:
                    mlflow.sklearn.log_model(
                        model, 
                        name,
                        registered_model_name=name
                    )
                    print(f"Modelo {name} registrado exitosamente")
                except Exception as e:
                    print(f"Error al registrar el modelo {name}: {e}")
                
                avg_score = (metrics['accuracy'] + metrics['f1_score']) / 2
                model_metrics[name] = avg_score
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = name
                    best_model_obj = model

                print(f"Modelo {name} - Métricas: {metrics}")

        if best_model:
            print(f"Mejor modelo: {best_model} con puntuación: {best_score}")
            
            try:
                versions = client.search_model_versions(f"name='{best_model}'")
                
                if versions and len(versions) > 0:
                    latest_version = max([int(v.version) for v in versions])
                    
                    client.transition_model_version_stage(
                        name=best_model, 
                        version=latest_version, 
                        stage="Production"
                    )
                    
                    for version in versions:
                        if (version.version != str(latest_version) and 
                            version.current_stage == "Production"):
                            client.transition_model_version_stage(
                                name=best_model,
                                version=version.version,
                                stage="Archived"
                            )
                    
                    print(f"Modelo {best_model} versión {latest_version} marcado como Producción")
                    
                    for model_name in models.keys():
                        if model_name != best_model:
                            other_versions = client.search_model_versions(f"name='{model_name}'")
                            for v in other_versions:
                                if v.current_stage == "Production":
                                    client.transition_model_version_stage(
                                        name=model_name,
                                        version=v.version,
                                        stage="Archived"
                                    )
                else:
                    print(f"No se encontraron versiones para el modelo {best_model}. Es posible que el registro no se haya completado.")
                    
                    try:
                        print(f"Intentando registrar {best_model} manualmente...")
                        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{best_model}"
                        registered_model = mlflow.register_model(model_uri, best_model)
                        print(f"Modelo registrado manualmente: {registered_model.name}, versión: {registered_model.version}")
                        
                        client.transition_model_version_stage(
                            name=best_model,
                            version=registered_model.version,
                            stage="Production"
                        )
                        print(f"Modelo {best_model} versión {registered_model.version} marcado como Producción")
                    except Exception as e:
                        print(f"Error al intentar registro manual: {e}")
                        print("Continuando sin marcar un modelo como Producción")
                
            except Exception as e:
                print(f"Error al cambiar el estado del modelo: {e}")
                print("Detalles del error:", str(e))
    
    return {
        'best_model': best_model,
        'best_score': best_score,
        'model_metrics': model_metrics
    }

def update_batch_status(**kwargs):
    """Actualizar estado del batch procesado"""
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    query_current_batch = """
    SELECT MIN(batch_id) as current_batch
    FROM clean_data.batch_info
    WHERE batch_id IS NOT NULL
    """
    current_batch = pg_hook.get_first(query_current_batch)[0]
    
    pg_hook.run(f"""
    UPDATE clean_data.batch_info 
    SET batch_id = NULL 
    WHERE batch_id = {current_batch}
    """)
    
    query_remaining_batches = """
    SELECT COUNT(*) 
    FROM clean_data.batch_info 
    WHERE batch_id IS NOT NULL
    """
    remaining_batches = pg_hook.get_first(query_remaining_batches)[0]
    
    print(f"Batch procesado: {current_batch}, Lotes restantes: {remaining_batches}")
    
    return remaining_batches > 0

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 4, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'diabetes_ml_pipeline',
    default_args=default_args,
    description='Pipeline de ML para predicción de readmisión hospitalaria',
    schedule_interval=timedelta(days=1, hours=2),  # Un día y dos hora
    catchup=False,
    max_active_runs=1
)

set_mlflow_tracking_task = PythonOperator(
    task_id='set_mlflow_tracking',
    python_callable=set_mlflow_tracking,
    dag=dag
)

start_pipeline = DummyOperator(
    task_id='start_pipeline',
    dag=dag
)

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    provide_context=True,
    dag=dag
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag
)

train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    provide_context=True,
    dag=dag
)

start_pipeline >> set_mlflow_tracking_task >> load_data_task
load_data_task >> preprocess_data_task >> train_models_task
train_models_task 