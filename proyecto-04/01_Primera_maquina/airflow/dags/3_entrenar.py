from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import mlflow.lightgbm
from sklearn.tree import DecisionTreeRegressor

from mlflow.tracking import MlflowClient
import boto3
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Evidently API nueva
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

# Configuración de variables de entorno
MLFLOW_TRACKING_URI = "http://10.43.101.175:30500"
MLFLOW_S3_ENDPOINT_URL = "http://10.43.101.175:30382"
AWS_ACCESS_KEY_ID = "adminuser"
AWS_SECRET_ACCESS_KEY = "securepassword123"
HOST_IP = "10.43.101.175"
bucket_name = "mlflow-artifacts"
object_key = "preprocessors/preprocessor.joblib"  # Path within the bucket

def set_mlflow_tracking(**kwargs):
    """Configurar tracking de MLflow"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("lightgbm_housing")
    
    # Configurar credenciales de almacenamiento
    os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = MLFLOW_S3_ENDPOINT_URL
    os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY

    print("✅ Tracking de MLflow configurado exitosamente")


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 5, 27),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
    'email_on_retry': False
}

dag = DAG(
    '3-Entrenar_modelo_con_drift_evidently',
    default_args=default_args,
    schedule_interval='5 0 * * *',
    catchup=False,
    max_active_runs=1,
    description='DAG para entrenar modelo con chequeo de data drift usando Evidently'
)

clean_schema = 'cleandata'
clean_table = 'processed_houses'
previous_data_path = '/tmp/previous_training_data.joblib'

def log_to_db(status, message, rmse=None):
    hook = PostgresHook(postgres_conn_id='postgres_default')
    sql = """
        INSERT INTO trainlogs.logs (status, message, rmse)
        VALUES (%s, %s, %s)
    """
    hook.run(sql, parameters=(status, message, rmse))

def check_data_count(**kwargs):
    hook = PostgresHook(postgres_conn_id='postgres_default')
    sql = f"SELECT COUNT(*) FROM {clean_schema}.{clean_table};"
    records = hook.get_first(sql)
    count = records[0] if records else 0
    print(f"Registros en tabla {clean_schema}.{clean_table}: {count}")
    if count > 20000:
        return "detect_data_drift_task"
    else:
        log_to_db("skip", "No se entrenó el modelo: menos de 20,000 registros.")
        return "skip_training_task"

def detect_data_drift(**kwargs):
    hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = hook.get_sqlalchemy_engine()
    query = f"SELECT * FROM {clean_schema}.{clean_table};"
    df_new = pd.read_sql(query, con=engine)

    relevant_cols = ['price', 'bed', 'bath', 'acre_lot']
    df_new = df_new[relevant_cols]

    categorical_cols = df_new.select_dtypes(include='object').columns.tolist()
    df_new[categorical_cols] = df_new[categorical_cols].astype('category')

    numerical_cols = [col for col in df_new.columns if col not in categorical_cols + ['price']]

    column_mapping = ColumnMapping(
        target='price',
        numerical_features=numerical_cols,
        categorical_features=categorical_cols
    )

    if not os.path.exists(previous_data_path):
        joblib.dump(df_new, previous_data_path)
        log_to_db("train", "No se detectó drift (primer entrenamiento).")
        return "train_model_task"

    df_old = joblib.load(previous_data_path)

    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=df_old, current_data=df_new, column_mapping=column_mapping)
    result = report.as_dict()

    drift_flag = result['metrics'][0]['result'].get('dataset_drift', False)
    target_drift_score = result['metrics'][1]['result'].get('drift_score', 0.0)

    if drift_flag or target_drift_score > 0.05:
        log_to_db("skip", f"Drift detectado, se reentrenara modelo: dataset_drift={drift_flag}")
        return "train_model_task"
    else:
        joblib.dump(df_new, previous_data_path)
        log_to_db("train", "No se detectó drift, no se procede a entrenar el modelo.")
        return "skip_training_task"
    
def preprocess_data(df, **kwargs):
    """Preprocesar datos de entrenamiento con nombres de columnas descriptivos"""
    # Objetivo (target)
    y_train = df['price']
    
    # Features
    X_train = df.drop(columns=['id', 'price', 'prev_sold_date', 'price_per_sqft','data_origin'], errors='ignore')
    
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Crear ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Ajustar y transformar
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Obtener nombres de columnas procesadas
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])]
    
    # Convertir a DataFrame para mantener nombres
    if hasattr(X_train_processed, 'toarray'):  # Si es sparse
        X_train_processed = X_train_processed.toarray()
    
    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    
    # Guardar el preprocesador localmente
    local_path = '/tmp/preprocessor.joblib'
    with open(local_path, 'wb') as f:
        joblib.dump(preprocessor, f)
    
    # Subir a MinIO
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=MLFLOW_S3_ENDPOINT_URL,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        print("✅ MinIO client initialized successfully.")
        s3_client.upload_file(local_path, bucket_name, object_key)
        print(f"✅ Preprocessor uploaded to MinIO at s3://{bucket_name}/{object_key}")
    except Exception as e:
        print(f"❌ Failed to upload preprocessor to MinIO: {e}")
    
    return X_train_processed, y_train

    
def train_model(**kwargs):
    hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = hook.get_sqlalchemy_engine()
    query = f"SELECT * FROM {clean_schema}.{clean_table};"
    df = pd.read_sql(query, con=engine)
    df = df[:5000]
    # Calcular el porcentaje de 'teacher' en la columna 'data_origin'
    porcentaje_teacher = (df['data_origin'] == 'teacher').mean() * 100

    # Imprimir el porcentaje
    print(f"Porcentaje de 'teacher' en la columna 'data_origin': {porcentaje_teacher:.2f}%")

    # Hacer la condición: ¿Es mayor al 80%?
    if porcentaje_teacher > 80:
        print("Más del 80% de los datos son 'teacher'")
    else:
        print("Menos del 80% de los datos son 'teacher'")
        return "skip_training_task"
    
    X, y = preprocess_data(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    joblib.dump(df, previous_data_path)
    print(f"Datos de entrenamiento guardados en {previous_data_path}")

    # Configurar MLflow
    set_mlflow_tracking()
    client = MlflowClient()

    # Modelos a entrenar
    modelos = {
        'LightGBMRegressor': LGBMRegressor(
            objective='regression',
            n_estimators=5,
            random_state=42,
            max_depth=3,
        ),
        'DecisionTreeRegressor': DecisionTreeRegressor(
            max_depth=3,
            random_state=42
        )
    }

    resultados = {}

    # Entrenamiento de modelos
    for model_name, model in modelos.items():
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            print(f"\nIniciando entrenamiento de {model_name} (run_id: {run_id})...")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            print(f"RMSE validación: {rmse:.4f}")

            if model_name == 'LightGBMRegressor':
                mlflow.log_param("n_estimators", 10)
                mlflow.log_param("max_depth", 5)
            elif model_name == 'DecisionTreeRegressor':
                mlflow.log_param("max_depth", 5)

            mlflow.log_metric("rmse", rmse)

            if model_name == 'LightGBMRegressor':
                mlflow.lightgbm.log_model(model, artifact_path="model", registered_model_name=model_name)
            else:
                mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_name)

            resultados[model_name] = {
                'rmse': rmse,
                'run_id': run_id
            }

    # Encontrar el mejor modelo de esta ejecución
    mejor_modelo = min(resultados.items(), key=lambda x: x[1]['rmse'])
    mejor_modelo_nombre = mejor_modelo[0]
    mejor_run_id = mejor_modelo[1]['run_id']
    mejor_rmse = mejor_modelo[1]['rmse']

    # Pasar a Production
    try:
        versions = client.search_model_versions(f"name='{mejor_modelo_nombre}'")
        latest_version = max(int(v.version) for v in versions)

        client.transition_model_version_stage(
            name=mejor_modelo_nombre,
            version=latest_version,
            stage="Production"
        )

        # Archivar los demás modelos (opcional)
        for nombre, data in resultados.items():
            if nombre != mejor_modelo_nombre:
                other_versions = client.search_model_versions(f"name='{nombre}'")
                for v in other_versions:
                    if v.current_stage != "Archived":
                        client.transition_model_version_stage(
                            name=nombre,
                            version=v.version,
                            stage="Archived"
                        )

        print(f"\n✅ Modelo {mejor_modelo_nombre} v{latest_version} marcado como Production (RMSE: {mejor_rmse:.4f})")
        log_to_db("stage", f"Modelo {mejor_modelo_nombre} v{latest_version} marcado como Production (RMSE {mejor_rmse:.4f})", rmse=mejor_rmse)

    except Exception as e:
        print(f"Error al marcar el modelo como Production: {e}")
        log_to_db("train", f"Error en proceso de transición a Production: {e}", rmse=mejor_rmse)

    return {
        'mejor_modelo': mejor_modelo_nombre,
        'rmse': mejor_rmse
    }


with dag:


    set_mlflow_tracking_task = PythonOperator(
        task_id='set_mlflow_tracking',
        python_callable=set_mlflow_tracking,
        dag=dag
    )

    check_data_task = BranchPythonOperator(
        task_id='check_data_count',
        python_callable=check_data_count
    )

    detect_data_drift_task = BranchPythonOperator(
        task_id='detect_data_drift_task',
        python_callable=detect_data_drift
    )

    train_model_task = PythonOperator(
        task_id='train_model_task',
        python_callable=train_model
    )

    skip_training_task = DummyOperator(
        task_id='skip_training_task'
    )

    end_task = DummyOperator(
        task_id='end_task',
        trigger_rule='none_failed_min_one_success'
    )

check_data_task >> detect_data_drift_task >> [set_mlflow_tracking_task >> train_model_task, skip_training_task] >> end_task
