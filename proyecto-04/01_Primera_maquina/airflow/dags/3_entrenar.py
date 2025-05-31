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
        log_to_db("skip", f"Drift detectado: dataset_drift={drift_flag}, target_drift_score={target_drift_score:.4f}")
        return "skip_training_task"
    else:
        joblib.dump(df_new, previous_data_path)
        log_to_db("train", "No se detectó drift, se procede a entrenar el modelo.")
        return "train_model_task"
    
def preprocess_data(df,**kwargs):
    """Preprocesar datos de entrenamiento con eficiencia de memoria mejorada"""
   
    # Cargar desde archivo temporal
    
    y_train = df['price'].tolist()
    X_train = df.drop(columns=['id', 'price', 'prev_sold_date'], errors='ignore')
    
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Realizar fit_transform una sola vez
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Guardar el preprocesador
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
 
   
    return X_train_processed,y_train
    



def train_model(**kwargs):
    hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = hook.get_sqlalchemy_engine()
    query = f"SELECT * FROM {clean_schema}.{clean_table};"
    df = pd.read_sql(query, con=engine)
    X,y = preprocess_data(df)

    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category')

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    df_train = X_train.copy()
    df_train['price'] = y_train
    joblib.dump(df_train, previous_data_path)
    print(f"Datos de entrenamiento guardados en {previous_data_path}")
    # Configurar MLflow
    set_mlflow_tracking()  # Define esta función para setear MLFLOW_TRACKING_URI, etc.
    client = MlflowClient()
    model_name = 'LightGBMRegressor'

    model = LGBMRegressor(
        objective='regression',
        n_estimators=50,
        random_state=42,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.7,
        n_jobs=1
    )
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Iniciando entrenamiento de {model_name} (run_id: {run_id})...")

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_val, num_iteration=model.best_iteration_)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(f"RMSE validación: {rmse:.4f}")

        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("rmse", rmse)
        mlflow.lightgbm.log_model(model, artifact_path="model", registered_model_name=model_name)

        # Verificar si el nuevo modelo es mejor que el actual en producción
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            production_versions = [v for v in versions if v.current_stage == "Production"]
            best_production_rmse = None

            if production_versions:
                prod_version = production_versions[0]
                prod_run_id = prod_version.run_id
                prod_metrics = client.get_run(prod_run_id).data.metrics
                best_production_rmse = prod_metrics.get('rmse', None)

            if best_production_rmse is None or rmse < best_production_rmse:
                latest_version = max(int(v.version) for v in versions)
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version,
                    stage="Production"
                )
                # Archivar versiones anteriores
                for v in versions:
                    if v.version != str(latest_version) and v.current_stage == "Production":
                        client.transition_model_version_stage(
                            name=model_name,
                            version=v.version,
                            stage="Archived"
                        )
                print(f"Modelo {model_name} v{latest_version} marcado como Producción")
                log_to_db("stage", f"Modelo {model_name} v{latest_version} marcado como Producción (RMSE {rmse:.4f})")
            else:
                best_production_rmse = prod_metrics.get('rmse', None)
                print(f"El nuevo modelo tiene un RMSE {rmse:.4f} mayor que el actual en producción ({best_production_rmse:.4f}). No se actualiza la producción.")
                log_to_db("train", f"Modelo no actualizado a Producción. RMSE actual: {rmse:.4f}, RMSE en Producción: {best_production_rmse:.4f}", rmse=rmse)

        except Exception as e:
            print(f"Error al comparar o cambiar el estado del modelo: {e}")
            log_to_db("train", f"Error en proceso de transición: {e}", rmse=rmse)

        print(f"Modelo {model_name} entrenado y logueado en MLflow.")

    return {
        'best_model': model_name,
        'rmse': rmse
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
