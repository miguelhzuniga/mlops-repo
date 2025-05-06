import os
import mlflow
import pandas as pd
import numpy as np
from mlflow.exceptions import MlflowException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import gradio as gr
import traceback
import requests
import json
from io import BytesIO
import boto3
import joblib
import dill  
import tempfile
import logging
import uvicorn
from fastapi import FastAPI
from threading import Thread


logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Prometheus metrics
REQUESTS = Counter('diabetes_gradio_requests_total', 'Número total de solicitudes a la interfaz Gradio')
PREDICTIONS = Counter('diabetes_gradio_predictions_total', 'Número total de predicciones realizadas')
PREDICTION_TIME = Histogram('diabetes_gradio_prediction_time_seconds', 
                           'Tiempo empleado en procesar solicitudes de predicción')
MODEL_ERRORS = Counter('diabetes_gradio_model_errors_total', 'Número total de errores del modelo')
PREPROCESSOR_ERRORS = Counter('diabetes_gradio_preprocessor_errors_total', 'Errores al cargar/aplicar el preprocesador')
MODEL_LOADS = Counter('diabetes_gradio_model_loads_total', 'Número de veces que se cargó un modelo')
REFRESH_CALLS = Counter('diabetes_gradio_refresh_calls_total', 'Número de veces que se actualizó la lista de modelos')

# Create FastAPI app for Prometheus metrics
metrics_app = FastAPI(title="Prometheus Metrics for Diabetes Gradio")

@metrics_app.get("/")
async def root():
    return {"message": "Prometheus Metrics Server for Diabetes Gradio"}

@metrics_app.get("/metrics")
async def metrics():
    return generate_latest(), {"Content-Type": CONTENT_TYPE_LATEST}

# Function to run metrics server in a separate thread
def run_metrics_server():
    uvicorn.run(metrics_app, host="0.0.0.0", port=9090)

# MLFlow Configuration
HOST_IP = "10.43.101.206"
os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{HOST_IP}:30382"
os.environ['AWS_ACCESS_KEY_ID'] = "adminuser"
os.environ['AWS_SECRET_ACCESS_KEY'] = "securepassword123"
mlflow.set_tracking_uri(f"http://{HOST_IP}:30500")

# MinIO Configuration
MINIO_ENDPOINT = f"http://{HOST_IP}:30382"
AWS_ACCESS_KEY = "adminuser"
AWS_SECRET_KEY = "securepassword123"
BUCKET_NAME = "mlflow-artifacts"
PREPROCESSOR_KEY = "preprocessors/preprocessor.joblib"

# Column configurations - from main_server.py
NUMERIC_COLUMNS = [
    'encounter_id', 'patient_nbr', 'admission_type_id', 
    'discharge_disposition_id', 'admission_source_id', 
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

CATEGORICAL_COLUMNS = [
    'race', 'gender', 'age', 'weight', 'payer_code', 
    'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 
    'max_glu_serum', 'A1Cresult', 'a1cresult', 'metformin', 
    'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 
    'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 
    'troglitazone', 'tolazamide', 'examide', 'citoglipton', 
    'insulin', 'glyburide-metformin', 'glipizide-metformin', 
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 
    'metformin-pioglitazone', 'change', 'diabetesMed', 'diabetesmed'
]

# Field mapping for names with hyphens
FIELD_MAPPING = {
    'glyburide_metformin': 'glyburide-metformin',
    'glipizide_metformin': 'glipizide-metformin',
    'glimepiride_pioglitazone': 'glimepiride-pioglitazone',
    'metformin_rosiglitazone': 'metformin-rosiglitazone',
    'metformin_pioglitazone': 'metformin-pioglitazone'
}

# Caches for preprocessor and model
preprocessor_cache = None
loaded_models = {}
current_model_name = None

# Initialize S3 client
s3_client = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

def store_prediction_in_postgres(input_dict, prediction_result):
    """Almacena los datos de predicción en PostgreSQL remoto"""
    try:
        import psycopg2
        
        # Configuración para la conexión a PostgreSQL de Airflow
        POSTGRES_HOST = "10.43.101.175"  # IP de la máquina con Airflow
        POSTGRES_PORT = 5432      # Puerto estándar de PostgreSQL 
        POSTGRES_DB = "airflow"   # Base de datos de Airflow
        POSTGRES_USER = "airflow" # Usuario de Airflow
        POSTGRES_PASSWORD = "airflow"  # Ajustar contraseña correcta
        
        # Conexión a PostgreSQL
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        
        cur = conn.cursor()
        
        input_dict['readmitted'] = prediction_result
        input_dict['dataset'] = 'prediction'
        
        columns = list(input_dict.keys())
        placeholders = ["%s"] * len(columns)
        values = [input_dict[col] for col in columns]
        
        query = f"""
        INSERT INTO raw_data.diabetes ({', '.join([f'"{col}"' for col in columns])})
        VALUES ({', '.join(placeholders)})
        """
        
        cur.execute(query, values)
        conn.commit()
        conn.close()
        
        logger.info("Predicción almacenada correctamente con dataset='prediction'")
        return True
    except Exception as e:
        logger.error(f"Error almacenando predicción: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    
def normalize_field_names(data_dict):
    """
    Normalizes field names, replacing underscores with hyphens
    when needed according to the defined mapping.
    """
    normalized = {}
    for key, value in data_dict.items():
        # Check if the field is in the mapping
        if key in FIELD_MAPPING:
            normalized_key = FIELD_MAPPING[key]
            logger.info(f"Normalizing field: {key} -> {normalized_key}")
        else:
            normalized_key = key
        normalized[normalized_key] = value
    return normalized

def load_preprocessor():
    """Load the preprocessor from MinIO using the same logic as in main_server.py"""
    global preprocessor_cache
    
    if preprocessor_cache is not None:
        logger.info("Using cached preprocessor")
        return preprocessor_cache
    
    try:
        logger.info(f"Downloading preprocessor from s3://{BUCKET_NAME}/{PREPROCESSOR_KEY}")
        
        # Create a temporary file to store the preprocessor
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_path = tmp_file.name
            
        # Download the preprocessor
        s3_client.download_file(BUCKET_NAME, PREPROCESSOR_KEY, temp_path)
        
        try:
            # Try loading with joblib first (preferred method according to the DAG)
            preprocessor_cache = joblib.load(temp_path)
            logger.info("Preprocessor loaded successfully with joblib")
        except Exception as joblib_error:
            logger.warning(f"Error loading with joblib: {joblib_error}. Trying with dill...")
            # If it fails, try with dill as a backup
            with open(temp_path, 'rb') as f:
                preprocessor_cache = dill.load(f)
            logger.info("Preprocessor loaded successfully with dill")
        
        # Delete the temporary file
        os.unlink(temp_path)
        
        # Verify that the preprocessor is as expected (ColumnTransformer)
        if hasattr(preprocessor_cache, 'transformers'):
            logger.info(f"Preprocessor verified. Transformers: {len(preprocessor_cache.transformers)}")
        else:
            logger.warning("Preprocessor doesn't appear to be a ColumnTransformer")
        
        return preprocessor_cache
    
    except Exception as e:
        PREPROCESSOR_ERRORS.inc()
        logger.error(f"Error loading the preprocessor: {str(e)}")
        raise

def list_mlflow_models():
    """List models from MLflow, focusing on production models"""
    REFRESH_CALLS.inc()
    try:
        client = mlflow.tracking.MlflowClient()
        registered_models = client.search_registered_models()
        
        production_models = []
        
        for model in registered_models:
            model_info = {
                "name": model.name,
                "versions": [],
                "lastUpdated": model.last_updated_timestamp
            }
            
            # Filter only Production versions
            for version in client.search_model_versions(f"name='{model.name}'"):
                if version.current_stage == "Production":
                    model_info["versions"].append({
                        "version": version.version,
                        "creation_timestamp": version.creation_timestamp
                    })
            
            if model_info["versions"]:  # Add model if it has Production versions
                production_models.append(model_info)
        
        return production_models
    
    except Exception as e:
        MODEL_ERRORS.inc()
        logger.error(f"Error listing production models: {e}")
        return {"success": False, "error": str(e)}

def load_model(model_name):
    """Load model with improved error handling from main_server.py"""
    global current_model_name, loaded_models
    
    if not model_name:
        return " Por favor, seleccione un modelo para cargar."
    
    try:
        if model_name in loaded_models:
            current_model_name = model_name
            logger.info(f"Modelo '{model_name}' ya cargado y seleccionado.")
            return f" Modelo '{model_name}' seleccionado."
        
        logger.info(f"=== INICIO DE CARGA DE MODELO: {model_name} ===")
        
        try:
            client = mlflow.tracking.MlflowClient()
            
            versions = client.search_model_versions(f"name='{model_name}'")
            production_versions = [v for v in versions if v.current_stage == "Production"]
            
            if not production_versions:
                return f" No se encontró una versión en producción para el modelo '{model_name}'."
            
            latest_prod_version = sorted(production_versions, key=lambda x: int(x.version), reverse=True)[0]
            
            logger.info(f"Cargando versión de producción: {latest_prod_version.version} para {model_name}")
            logger.info(f"Run ID: {latest_prod_version.run_id}")
            logger.info(f"Source: {latest_prod_version.source}")
            
            model_uri = f"models:/{model_name}/Production"
            logger.info(f"Intentando cargar modelo con URI: {model_uri}")
            
            # Load the model directly
            loaded_models[model_name] = mlflow.pyfunc.load_model(model_uri)
            current_model_name = model_name
            
            MODEL_LOADS.inc()
            logger.info(f"Modelo '{model_name}' cargado exitosamente")
                        
            return f" Modelo '{model_name}' cargado y seleccionado correctamente."
            
        except Exception as e:
            MODEL_ERRORS.inc()
            logger.error(f"Error al cargar modelo: {type(e).__name__}: {str(e)}")
            logger.error(f"Traza completa: {traceback.format_exc()}")
            
            try:
                logger.info("Intentando método de carga alternativo por nombre/etapa...")
                model_uri = f"models:/{model_name}/Production"
                loaded_models[model_name] = mlflow.pyfunc.load_model(model_uri)
                current_model_name = model_name
                MODEL_LOADS.inc()
                return f" Modelo '{model_name}' cargado y seleccionado correctamente con método alternativo."
            except Exception as alt_error:
                MODEL_ERRORS.inc()
                logger.error(f"Error en método alternativo: {str(alt_error)}")
            
            if "404" in str(e) or "Not Found" in str(e):
                return f" No se pudo cargar el modelo '{model_name}'. No se encontró el archivo del modelo en el almacenamiento S3. Verifique la ruta y las credenciales de S3."
            else:
                return f" No se pudo cargar el modelo '{model_name}'. Error: {str(e)}"
    except Exception as e:
        MODEL_ERRORS.inc()
        logger.error(f"=== ERROR DE CARGA DE MODELO ===")
        logger.error(f"Error detallado: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
        
        return f"""
        <div style="background-color: #ffebee; padding: 15px; border-radius: 10px; border-left: 5px solid #f44336;">
            <h3 style="color: #c62828; margin-top: 0;"> Error al cargar el modelo</h3>
            <p><strong>Descripción:</strong> {str(e)}</p>
            <p><strong>Tipo de error:</strong> {type(e).__name__}</p>
            <details>
                <summary>Detalles técnicos</summary>
                <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 12px;">
{traceback.format_exc()}
                </pre>
            </details>
        </div>
        """

def preprocess_input(input_data):
    """Preprocess input data with robust error handling from main_server.py"""
    try:
        logger.info("Iniciando preprocesamiento de datos")
        logger.info(f"Columnas recibidas: {input_data.columns.tolist()}")
        
        # Convert field names with underscores to hyphens
        for old_col, new_col in FIELD_MAPPING.items():
            if old_col in input_data.columns and new_col not in input_data.columns:
                input_data[new_col] = input_data[old_col]
                input_data = input_data.drop(old_col, axis=1)
                logger.info(f"Convertido campo {old_col} a {new_col}")
        
        # 1. For numeric columns, ensure they are float or int
        for col in NUMERIC_COLUMNS:
            if col in input_data.columns:
                # Try to convert to numeric, coerce errors to NaN
                try:
                    input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
                    # Replace NaN with 0
                    input_data[col] = input_data[col].fillna(0).astype(float)
                except Exception as e:
                    logger.warning(f"Error al convertir columna {col} a numérico: {e}")
                    input_data[col] = 0.0
            else:
                # If the column doesn't exist, add it with default values
                input_data[col] = 0.0
        
        # 2. For categorical columns, ensure they are strings
        for col in CATEGORICAL_COLUMNS:
            if col in input_data.columns:
                # Convert to string and replace null/empty values
                input_data[col] = input_data[col].astype(str)
                input_data[col] = input_data[col].replace(['?', 'nan', 'None', 'NULL', 'NaN', ''], 'Unknown')
                
                # Special handling for values with comparators
                if col in ['max_glu_serum', 'A1Cresult']:
                    # Don't attempt numeric conversions for these fields, always treat as categorical
                    if input_data[col].str.contains('>').any() or input_data[col].str.contains('<').any():
                        logger.info(f"Detectados valores con comparadores en columna {col}")
            else:
                # If the column doesn't exist, add it with default values
                if col in ['insulin', 'metformin', 'change'] or col.endswith('metformin') or '-' in col:
                    input_data[col] = 'No'  # For medications
                else:
                    input_data[col] = 'Unknown'  # For other categories
        
        # 3. Handle duplicates with different capitalization
        if 'A1Cresult' in input_data.columns:
            input_data['a1cresult'] = input_data['A1Cresult']
        if 'diabetesMed' in input_data.columns:
            input_data['diabetesmed'] = input_data['diabetesMed']
        
        # 4. Remove the readmitted column if present
        if "readmitted" in input_data.columns:
            input_data = input_data.drop("readmitted", axis=1)
        
        # Verify before applying the preprocessor
        logger.info("Tipos de datos antes del preprocesamiento:")
        for col in input_data.columns:
            logger.info(f"  {col}: {input_data[col].dtype}")
        
        # Load and apply the preprocessor
        preprocessor = load_preprocessor()
        
        # Test conversion: Verify the DataFrame can be correctly converted
        try:
            dummy_array = input_data.to_numpy()
            logger.info(f"Conversión a numpy exitosa: {dummy_array.shape}")
        except Exception as e:
            logger.error(f"Error al convertir DataFrame a numpy: {e}")
            # Try to repair each column individually
            for col in input_data.columns:
                try:
                    dummy = input_data[col].to_numpy()
                except Exception as col_e:
                    logger.error(f"Error en columna {col}: {col_e}")
                    if col in NUMERIC_COLUMNS:
                        input_data[col] = 0.0
                    else:
                        input_data[col] = 'Unknown'
        
        # Apply the preprocessor - wrapped in an exception handler
        try:
            processed_data = preprocessor.transform(input_data)
            logger.info(f"Preprocesamiento exitoso: {processed_data.shape}")
            return processed_data
        except Exception as transform_error:
            logger.error(f"Error en transform: {transform_error}")
            # Recovery attempt: use only columns matching the DAG
            subset_columns = [col for col in input_data.columns if col in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
            input_subset = input_data[subset_columns].copy()
            
            # Verify and reapply types
            for col in subset_columns:
                if col in NUMERIC_COLUMNS:
                    input_subset[col] = input_subset[col].astype(float)
                else:
                    input_subset[col] = input_subset[col].astype(str)
            
            # Try again with the subset
            logger.info(f"Intentando con subconjunto de {len(subset_columns)} columnas")
            processed_data = preprocessor.transform(input_subset)
            return processed_data
    
    except Exception as e:
        PREPROCESSOR_ERRORS.inc()
        logger.error(f"Error en el preprocesamiento: {str(e)}")
        logger.error(f"Columnas en los datos: {input_data.columns.tolist()}")
        raise Exception(f"Error de preprocesamiento: {str(e)}")
        
def predict(
    model_name,
    race,
    gender,
    age,
    weight,
    payer_code,
    medical_specialty,
    admission_type_id,
    discharge_disposition_id,
    admission_source_id,
    time_in_hospital,
    num_lab_procedures,
    num_procedures,
    num_medications,
    number_outpatient,
    number_emergency,
    number_inpatient,
    number_diagnoses,
    diag_1,
    diag_2,
    diag_3,
    max_glu_serum,
    A1Cresult,
    metformin,
    repaglinide,
    nateglinide,
    chlorpropamide,
    glimepiride,
    acetohexamide,
    glipizide,
    glyburide,
    tolbutamide,
    pioglitazone,
    rosiglitazone,
    acarbose,
    miglitol,
    troglitazone,
    tolazamide,
    examide,
    citoglipton,
    insulin,
    glyburide_metformin,
    glipizide_metformin,
    glimepiride_pioglitazone,
    metformin_rosiglitazone,
    metformin_pioglitazone,
    change,
    diabetesMed
):
    """Make prediction with improved preprocessing from main_server.py"""
    REQUESTS.inc()
    global current_model_name, loaded_models
    
    with PREDICTION_TIME.time():
        if not current_model_name and not model_name:
            return " No hay un modelo seleccionado. Por favor, seleccione un modelo antes de realizar predicciones."
        
        model_to_use = model_name if model_name else current_model_name
        
        if model_to_use not in loaded_models:
            return f" El modelo '{model_to_use}' no está cargado. Por favor, selecciónelo primero."
        
        try:
            # Create input data dictionary and normalize field names
            input_dict = {
                'race': race,
                'gender': gender,
                'age': age,
                'weight': weight if weight else 'None',
                'payer_code': payer_code if payer_code else 'Unknown',
                'medical_specialty': medical_specialty if medical_specialty else 'Unknown',
                'admission_type_id': int(admission_type_id),
                'discharge_disposition_id': int(discharge_disposition_id),
                'admission_source_id': int(admission_source_id),
                'time_in_hospital': int(time_in_hospital),
                'num_lab_procedures': int(num_lab_procedures),
                'num_procedures': int(num_procedures),
                'num_medications': int(num_medications),
                'number_outpatient': int(number_outpatient),
                'number_emergency': int(number_emergency),
                'number_inpatient': int(number_inpatient),
                'number_diagnoses': int(number_diagnoses),
                'diag_1': diag_1 if diag_1 else '250.00',
                'diag_2': diag_2 if diag_2 else '250.00',
                'diag_3': diag_3 if diag_3 else '250.00',
                'max_glu_serum': max_glu_serum,
                'A1Cresult': A1Cresult,
                'metformin': metformin,
                'repaglinide': repaglinide,
                'nateglinide': nateglinide,
                'chlorpropamide': chlorpropamide,
                'glimepiride': glimepiride,
                'acetohexamide': acetohexamide,
                'glipizide': glipizide,
                'glyburide': glyburide,
                'tolbutamide': tolbutamide,
                'pioglitazone': pioglitazone,
                'rosiglitazone': rosiglitazone,
                'acarbose': acarbose,
                'miglitol': miglitol,
                'troglitazone': troglitazone,
                'tolazamide': tolazamide,
                'examide': examide,
                'citoglipton': citoglipton,
                'insulin': insulin,
                'glyburide_metformin': glyburide_metformin,
                'glipizide_metformin': glipizide_metformin,
                'glimepiride_pioglitazone': glimepiride_pioglitazone,
                'metformin_rosiglitazone': metformin_rosiglitazone,
                'metformin_pioglitazone': metformin_pioglitazone,
                'change': change if change else 'No',
                'diabetesMed': diabetesMed
            }
            
            # Normalize field names using the logic from main_server.py
            normalized_data = normalize_field_names(input_dict)
            
            # Convert to DataFrame
            input_data = pd.DataFrame([normalized_data])
            logger.info(f"Datos de entrada creados con {len(input_data.columns)} columnas")
            
            # Apply robust preprocessing from main_server.py
            input_data_processed = preprocess_input(input_data)
            logger.info(f"Datos preprocesados exitosamente con forma {input_data_processed.shape}")
            
            # Get the model and make prediction
            model = loaded_models[model_to_use]
            prediction = model.predict(input_data_processed)
            PREDICTIONS.inc()
            logger.info(f"Predicción realizada: {prediction}")
            
            # COMENTADO: Almacenamiento en PostgreSQL
            # storage_result = store_prediction_in_postgres(normalized_data, prediction[0])
            # storage_message = "Datos almacenados correctamente." if storage_result else "No se pudieron almacenar los datos."
            
            readmitted_types = {
                "NO": "No readmitido",
                "<30": "Readmitido en menos de 30 días",
                ">30": "Readmitido después de 30 días"
            }
            
            try:
                pred_value = str(prediction[0])
                readmitted_type = readmitted_types.get(pred_value, f"Tipo {pred_value}")
            except Exception as inner_e:
                logger.error(f"Error al interpretar el resultado: {str(inner_e)}")
                readmitted_type = str(prediction[0])
            
            result = f"""
            <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50;">
                <h3 style="color: #2e7d32; margin-top: 0;"> Resultado de la Predicción</h3>
                <div style="font-size: 18px; margin-bottom: 15px;">
                    <strong>Readmisión del paciente:</strong> <span style="background-color: #81c784; padding: 5px 10px; border-radius: 5px; color: white;">{readmitted_type} ({prediction[0]})</span>
                </div>
                <div style="font-size: 16px; margin-bottom: 15px;">
                    <strong>Modelo utilizado:</strong> {model_to_use}
                </div>
                
                <div style="margin-top: 20px;">
                    <h4 style="color: #2e7d32;"> Resumen de datos de entrada</h4>
                    <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                        <tr style="background-color: #c8e6c9;">
                            <th style="padding: 8px; text-align: left; border: 1px solid #a5d6a7;">Característica</th>
                            <th style="padding: 8px; text-align: left; border: 1px solid #a5d6a7;">Valor</th>
                        </tr>
            """
            
            # Create a more focused view of key input parameters
            input_data_display = {
                "Raza": race,
                "Género": gender,
                "Grupo de edad": age,
                "Tipo de admisión": admission_type_id,
                "Tiempo en hospital (días)": time_in_hospital,
                "Procedimientos de laboratorio": num_lab_procedures,
                "Procedimientos": num_procedures,
                "Número de medicamentos": num_medications,
                "Visitas ambulatorias": number_outpatient,
                "Visitas a emergencia": number_emergency,
                "Hospitalizaciones previas": number_inpatient,
                "Número de diagnósticos": number_diagnoses,
                "Nivel máximo de glucosa sérica": max_glu_serum,
                "Resultado de HbA1c": A1Cresult,
                "Cambio en dosis de insulina": insulin,
                "Medicamento para diabetes": diabetesMed
            }
            
            for key, value in input_data_display.items():
                result += f"""
                        <tr>
                            <td style="padding: 8px; border: 1px solid #a5d6a7;">{key}</td>
                            <td style="padding: 8px; border: 1px solid #a5d6a7;">{value}</td>
                        </tr>
                """
            
            result += """
                    </table>
                </div>
                
                <div style="margin-top: 20px; font-size: 14px; color: #555;">
                    <p><strong>Nota:</strong> Esta predicción se basa en las variables de entrada proporcionadas. Para un análisis detallado, consulte con profesionales de la salud.</p>
                </div>
            </div>
            """
            
            return result
        
        except Exception as e:
            MODEL_ERRORS.inc()
            logger.error(f"=== ERROR EN PREDICCIÓN ===")
            logger.error(f"Error detallado: {type(e).__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            
            return f"""
            <div style="background-color: #ffebee; padding: 15px; border-radius: 10px; border-left: 5px solid #f44336;">
                <h3 style="color: #c62828; margin-top: 0;"> Error en la predicción</h3>
                <p><strong>Descripción:</strong> {str(e)}</p>
                <p><strong>Tipo de error:</strong> {type(e).__name__}</p>
                <details>
                    <summary>Detalles técnicos</summary>
                    <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 12px;">
{traceback.format_exc()}
                    </pre>
                </details>
                <p><strong>Recomendación:</strong> Verifique que el formato de los datos de entrada coincida con el formato esperado por el modelo. Si el problema persiste, considere reentrenar el modelo con un preprocesamiento más robusto.</p>
            </div>
            """

        
        except Exception as e:
            MODEL_ERRORS.inc()
            logger.error(f"=== ERROR EN PREDICCIÓN ===")
            logger.error(f"Error detallado: {type(e).__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            
            return f"""
            <div style="background-color: #ffebee; padding: 15px; border-radius: 10px; border-left: 5px solid #f44336;">
                <h3 style="color: #c62828; margin-top: 0;"> Error en la predicción</h3>
                <p><strong>Descripción:</strong> {str(e)}</p>
                <p><strong>Tipo de error:</strong> {type(e).__name__}</p>
                <details>
                    <summary>Detalles técnicos</summary>
                    <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 12px;">
{traceback.format_exc()}
                    </pre>
                </details>
                <p><strong>Recomendación:</strong> Verifique que el formato de los datos de entrada coincida con el formato esperado por el modelo. Si el problema persiste, considere reentrenar el modelo con un preprocesamiento más robusto.</p>
            </div>
            """
            
def refresh_models():
    """Refresh model list with improved logic"""
    REFRESH_CALLS.inc()
    models = list_mlflow_models()
    model_names = []
    model_info = ""

    if models:
        model_info = """
        <div style="background-color: #e8f0fe; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="color: #1976d2; margin-top: 0;">Modelos Disponibles en Producción</h3>
        """
        
        for model in models:
            # Add model name to list
            model_names.append(model["name"])
            
            production_badge = """ <span style="background-color: #4caf50; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px;">PRODUCCIÓN</span>"""
            versions_count = len(model["versions"]) if "versions" in model else 0
            
            model_info += f"""
            <div style="background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #1976d2;">
                <h4 style="margin-top: 0; margin-bottom: 10px; color: #1976d2;">{model["name"]}{production_badge}</h4>
                <p style="margin: 5px 0;"><strong>Versiones:</strong> {versions_count}</p>
            </div>
            """
        
        model_info += """
        </div>
        """
    else:
        model_info = """
        <div style="background-color: #fff3e0; padding: 15px; border-radius: 10px; border-left: 5px solid #ff9800;">
            <h3 style="color: #e65100; margin-top: 0;"> Sin Modelos Disponibles en Producción</h3>
            <p>No se encontraron modelos registrados en el estado de Producción en MLflow. Verifique la conexión con el servidor MLflow.</p>
        </div>
        """
    
    # Use update() to correctly update the dropdown
    return gr.Dropdown(choices=model_names, value=None if not model_names else model_names[0]), model_info

# Define Gradio theme (keep existing theme)
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont("Roboto"), "ui-sans-serif", "system-ui", "sans-serif"],
    spacing_size=gr.themes.sizes.spacing_md,
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#f9f9f9",
    body_background_fill_dark="#1a1a1a",
    button_primary_background_fill="#1976d2",
    button_primary_background_fill_hover="#1565c0",
    button_primary_text_color="white",
    button_secondary_background_fill="#e3f2fd",
    button_secondary_background_fill_hover="#bbdefb",
    button_secondary_text_color="#1976d2",
    block_title_text_color="#1976d2",
    block_label_text_color="#555",
    input_background_fill="#fff",
    input_border_color="#ddd",
    input_shadow="0 2px 4px rgba(0,0,0,0.05)",
    checkbox_background_color="#2196f3",
    slider_color="#2196f3",
    slider_color_dark="#64b5f6",
)

# Images for visual design
header_html = """
<div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
    <img src="https://cdn-icons-png.flaticon.com/512/2966/2966327.png" height="60px" style="margin-right: 20px;">
    <div>
        <h1 style="margin: 0; color: #1976d2; font-size: 28px;">Predictor de Readmisión de Pacientes Diabéticos</h1>
        <p style="margin: 5px 0 0; color: #555; font-size: 16px;">Modelo de aprendizaje automático para hospitales y centros de salud</p>
    </div>
</div>
"""

footer_html = """
<div style="margin-top: 30px; text-align: center; border-top: 1px solid #ddd; padding-top: 20px;">
    <p style="color: #555; font-size: 14px;">Sistema de Predicción de Readmisión Hospitalaria con MLflow y Gradio © 2025</p>
    <p style="color: #777; font-size: 12px;">Desarrollado para la mejora de la atención médica y gestión de pacientes diabéticos</p>
    <div style="display: flex; justify-content: center; gap: 15px; margin-top: 10px;">
        <div style="display: flex; align-items: center;">
            <span style="background-color: #2196f3; color: white; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></span>
            <span style="margin-left: 5px; color: #555; font-size: 12px;">MLflow</span>
        </div>
        <div style="display: flex; align-items: center;">
            <span style="background-color: #673ab7; color: white; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></span>
            <span style="margin-left: 5px; color: #555; font-size: 12px;">Gradio</span>
        </div>
        <div style="display: flex; align-items: center;">
            <span style="background-color: #ff9800; color: white; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></span>
            <span style="margin-left: 5px; color: #555; font-size: 12px;">Python</span>
        </div>
        <div style="display: flex; align-items: center;">
            <span style="background-color: #4caf50; color: white; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></span>
            <span style="margin-left: 5px; color: #555; font-size: 12px;">Prometheus</span>
        </div>
    </div>
</div>
"""

# Configuración de la aplicación Gradio con el tema personalizado
with gr.Blocks(theme=theme) as app:
    # Encabezado
    gr.HTML(header_html)
    
    # Panel de selección de modelo
    with gr.Tab("1️⃣ Selección de Modelo"):
        gr.Markdown("### Seleccione un modelo en Producción")
        
        with gr.Row():
            with gr.Column(scale=3):
                # Refresh button to map models
                refresh_button = gr.Button(" Mapear Modelos", variant="primary", size="lg")
                model_info = gr.HTML("*Haga clic en 'Mapear Modelos' para ver los modelos disponibles.*")
            
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("###  Carga de Modelo")
                    model_dropdown = gr.Dropdown(
                        label="Seleccione un modelo",
                        choices=[],  # Initially empty, will be updated after refresh
                        interactive=True
                    )
                    load_button = gr.Button(" Cargar Modelo", variant="primary")
                    load_output = gr.HTML()

        # Link the refresh button to fetch models and populate the dropdown
        refresh_button.click(refresh_models, outputs=[model_dropdown, model_info])

        # Link the load button to load the selected model
        load_button.click(load_model, inputs=model_dropdown, outputs=load_output)


    # Panel de predicción
    with gr.Tab("2️⃣ Realizar Predicción"):
        gr.Markdown("### Ingrese los datos del paciente para la predicción")
        
        # Usar pestañas para organizar los parámetros de entrada de manera más eficiente
        with gr.Tabs():
            with gr.TabItem(" Información Básica"):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("####  Información Demográfica")
                            race = gr.Dropdown(
                                label="Raza",
                                choices=["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"],
                                value="Caucasian",
                                info="Grupo étnico del paciente"
                            )
                            
                            gender = gr.Dropdown(
                                label="Género",
                                choices=["Male", "Female", "Unknown"],
                                value="Male",
                                info="Género del paciente"
                            )
                            
                            age = gr.Dropdown(
                                label="Grupo de edad",
                                choices=["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", 
                                         "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
                                value="[50-60)",
                                info="Rango de edad del paciente"
                            )
                            
                            weight = gr.Dropdown(
                                label="Peso",
                                choices=["[0-25)", "[25-50)", "[50-75)", "[75-100)", "[100-125)", "[125-150)", "[150-175)", "[175-200)", ">200"],
                                value="[75-100)",
                                info="Rango de peso del paciente"
                            )
                            
                            payer_code = gr.Dropdown(
                                label="Código de aseguradora",
                                choices=["Unknown", "MC", "MD", "HM", "UN", "BC", "CM", "CP", "CH", "SI", "SP", "WC", "OG", "OT", "PO", "DM", "FR"],
                                value="Unknown",
                                info="Código del proveedor de seguros"
                            )
                            
                            medical_specialty = gr.Dropdown(
                                label="Especialidad médica",
                                choices=["Unknown", "InternalMedicine", "Cardiology", "Emergency/Trauma", "Orthopedics", "Endocrinology", "Family/GeneralPractice", "Surgery-General", "Surgery-Neuro", "Surgery-Vascular", "Surgery-Thoracic", "Nephrology", "Gastroenterology", "Pulmonology", "Neurology", "Radiologist", "Psychiatry", "Urology", "ObstetricsAndGynecology"],
                                value="Unknown",
                                info="Especialidad médica del médico admitente"
                            )
                    
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("####  Información de Admisión")
                            admission_type_id = gr.Slider(
                                label="Tipo de admisión",
                                minimum=1,
                                maximum=8,
                                value=1,
                                step=1,
                                info="Tipo de admisión hospitalaria (1=Emergencia, 2=Urgente, 3=Electiva, etc.)"
                            )
                            
                            discharge_disposition_id = gr.Slider(
                                label="Disposición al alta",
                                minimum=1,
                                maximum=28,
                                value=1,
                                step=1,
                                info="Disposición del paciente al alta (1=Alta a domicilio, etc.)"
                            )
                            
                            admission_source_id = gr.Slider(
                                label="Fuente de admisión",
                                minimum=1,
                                maximum=25,
                                value=1,
                                step=1,
                                info="Fuente de admisión (1=Referencia médica, etc.)"
                            )
                            
                            time_in_hospital = gr.Slider(
                                label="Tiempo en hospital (días)",
                                minimum=1,
                                maximum=14,
                                value=4,
                                step=1,
                                info="Duración de la estadía hospitalaria"
                            )
            
            with gr.TabItem(" Historial Médico"):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("#### 離 Procedimientos y Medicamentos")
                            num_lab_procedures = gr.Slider(
                                label="Número de procedimientos de laboratorio",
                                minimum=1,
                                maximum=120,
                                value=45,
                                step=1,
                                info="Número de pruebas de laboratorio realizadas"
                            )
                            
                            num_procedures = gr.Slider(
                                label="Número de procedimientos",
                                minimum=0,
                                maximum=6,
                                value=1,
                                step=1,
                                info="Número de procedimientos (no de laboratorio) realizados"
                            )
                            
                            num_medications = gr.Slider(
                                label="Número de medicamentos",
                                minimum=1,
                                maximum=81,
                                value=16,
                                step=1,
                                info="Número total de medicamentos administrados"
                            )
                            
                            number_outpatient = gr.Slider(
                                label="Visitas ambulatorias",
                                minimum=0,
                                maximum=42,
                                value=0,
                                step=1,
                                info="Número de visitas ambulatorias en el año anterior"
                            )
                            
                            number_emergency = gr.Slider(
                                label="Visitas a emergencia",
                                minimum=0,
                                maximum=76,
                                value=0,
                                step=1,
                                info="Número de visitas a emergencia en el año anterior"
                            )
                                                    
                            number_inpatient = gr.Slider(
                                label="Hospitalizaciones previas",
                                minimum=0,
                                maximum=21,
                                value=0,
                                step=1,
                                info="Número de hospitalizaciones en el año anterior"
                            )
                            
                            number_diagnoses = gr.Slider(
                                label="Número de diagnósticos",
                                minimum=1,
                                maximum=16,
                                value=7,
                                step=1,
                                info="Número de diagnósticos registrados durante esta hospitalización"
                            )
                            
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("####  Diagnósticos")
                            diag_1 = gr.Textbox(
                                label="Diagnóstico primario",
                                value="250.00",
                                placeholder="Ingrese código ICD-9 (ej., 250.00)",
                                info="Diagnóstico primario (código ICD-9)"
                            )
                            
                            diag_2 = gr.Textbox(
                                label="Diagnóstico secundario",
                                value="250.00",
                                placeholder="Ingrese código ICD-9 (ej., 250.00)",
                                info="Diagnóstico secundario (código ICD-9)"
                            )
                            
                            diag_3 = gr.Textbox(
                                label="Diagnóstico adicional",
                                value="250.00",
                                placeholder="Ingrese código ICD-9 (ej., 250.00)",
                                info="Diagnóstico adicional (código ICD-9)"
                            )
                            
                            max_glu_serum = gr.Dropdown(
                                label="Nivel máximo de glucosa sérica",
                                choices=["None", "Norm", ">200", ">300"],
                                value="None",
                                info="Resultado de la prueba de glucosa sérica"
                            )
                            
                            A1Cresult = gr.Dropdown(
                                label="Resultado de HbA1c",
                                choices=["None", "Norm", ">7", ">8"],
                                value="None",
                                info="Resultado de la prueba de hemoglobina A1c"
                            )
            
            with gr.TabItem(" Medicamentos"):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("####  Insulina y Medicamentos para Diabetes")
                            insulin = gr.Dropdown(
                                label="Cambio en dosis de insulina",
                                choices=["No", "Up", "Down", "Steady"],
                                value="No",
                                info="Cambio en la dosis de insulina durante la hospitalización"
                            )
                            
                            diabetesMed = gr.Dropdown(
                                label="¿Se prescribió medicamento para diabetes?",
                                choices=["Yes", "No"],
                                value="Yes",
                                info="Si se prescribió algún medicamento para diabetes"
                            )
                            
                            change = gr.Dropdown(
                                label="Cambio en medicamentos para diabetes",
                                choices=["No", "Ch"],
                                value="No",
                                info="¿Hubo un cambio en los medicamentos para diabetes?"
                            )
                            
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("####  Medicamentos Orales (1)")
                            metformin = gr.Dropdown(
                                label="Metformina",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de metformina"
                            )
                            
                            repaglinide = gr.Dropdown(
                                label="Repaglinida",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de repaglinida"
                            )
                            
                            nateglinide = gr.Dropdown(
                                label="Nateglinida",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de nateglinida"
                            )
                            
                            chlorpropamide = gr.Dropdown(
                                label="Clorpropamida",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de clorpropamida"
                            )
                            
                            glimepiride = gr.Dropdown(
                                label="Glimepirida",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de glimepirida"
                            )
                            
                            acetohexamide = gr.Dropdown(
                                label="Acetohexamida",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de acetohexamida"
                            )
                            
                            glipizide = gr.Dropdown(
                                label="Glipizida",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de glipizida"
                            )
                    
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("####  Medicamentos Orales (2)")
                            glyburide = gr.Dropdown(
                                label="Gliburida",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de gliburida"
                            )
                            
                            tolbutamide = gr.Dropdown(
                                label="Tolbutamida",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de tolbutamida"
                            )
                            
                            pioglitazone = gr.Dropdown(
                                label="Pioglitazona",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de pioglitazona"
                            )
                            
                            rosiglitazone = gr.Dropdown(
                                label="Rosiglitazona",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de rosiglitazona"
                            )
                            
                            acarbose = gr.Dropdown(
                                label="Acarbosa",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de acarbosa"
                            )
                            
                            miglitol = gr.Dropdown(
                                label="Miglitol",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de miglitol"
                            )
                            
                            troglitazone = gr.Dropdown(
                                label="Troglitazona",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de troglitazona"
                            )
                            
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("####  Medicamentos Adicionales")
                            tolazamide = gr.Dropdown(
                                label="Tolazamida",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de tolazamida"
                            )
                            
                            examide = gr.Dropdown(
                                label="Examida",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de examida"
                            )
                            
                            citoglipton = gr.Dropdown(
                                label="Citogliptón",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de citogliptón"
                            )
                    
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("####  Medicamentos Combinados")
                            glyburide_metformin = gr.Dropdown(
                                label="Gliburida-Metformina",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de gliburida-metformina"
                            )
                            
                            glipizide_metformin = gr.Dropdown(
                                label="Glipizida-Metformina",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de glipizida-metformina"
                            )
                            
                            glimepiride_pioglitazone = gr.Dropdown(
                                label="Glimepirida-Pioglitazona",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de glimepirida-pioglitazona"
                            )
                            
                            metformin_rosiglitazone = gr.Dropdown(
                                label="Metformina-Rosiglitazona",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de metformina-rosiglitazona"
                            )
                            
                            metformin_pioglitazone = gr.Dropdown(
                                label="Metformina-Pioglitazona",
                                choices=["No", "Down", "Steady", "Up"],
                                value="No",
                                info="Estado de prescripción de metformina-pioglitazona"
                            )
        
        # Panel de resultados
        with gr.Row():
            predict_button = gr.Button(" Realizar Predicción", variant="primary", size="lg")
        
        with gr.Row():
            prediction_output = gr.HTML()
    
    # Panel de información
    with gr.Tab("ℹ️ Información"):
        gr.Markdown("""
        # Acerca del Predictor de Readmisión de Pacientes Diabéticos
        
        Esta aplicación utiliza modelos de aprendizaje automático para predecir la probabilidad de readmisión hospitalaria de pacientes diabéticos dentro de los 30 días posteriores al alta.
        
        ## Tipos de Resultado
        
        Los modelos pueden predecir tres tipos de resultados:
        
        1. **NO** - El paciente no será readmitido
        2. **<30** - El paciente será readmitido en menos de 30 días
        3. **>30** - El paciente será readmitido después de 30 días
        
        ## Variables predictoras
        
        Las variables utilizadas para la predicción corresponden a datos demográficos y clínicos:
        
        - **Datos demográficos** - Raza, género y grupo de edad
        - **Información de admisión** - Tipo de admisión, tiempo de estadía
        - **Procedimientos médicos** - Número de procedimientos de laboratorio y no laboratorio
        - **Medicamentos** - Número total de medicamentos administrados
        - **Historial de visitas** - Visitas ambulatorias, de emergencia y hospitalizaciones previas
        - **Información sobre diabetes** - Resultados de pruebas de glucosa, HbA1c y tratamiento con insulina
        
        ## Conjunto de datos
        
        Este proyecto está basado en el conjunto de datos de variables clinicas. El conjunto contiene más de 50 características que representan los resultados del paciente y del hospital.
        
        ## Desarrollo y tecnologías
        
        Esta aplicación está desarrollada con:
        - **MLflow** - Para la gestión y despliegue de modelos
        - **Gradio** - Para la interfaz de usuario
        - **Python** - Como lenguaje de programación base
        - **Kubernetes** - Para la orquestación de contenedores
        - **AirFlow** - Para la orquestación de flujos de trabajo
        - **PostgreSQL** - Para el almacenamiento de datos
        - **Prometheus** - Para el monitoreo de métricas
        - **FastAPI** - Para el desarrollo de APIs
        """)


    # Pie de página
    gr.HTML(footer_html)
    
    predict_button.click(
        fn=predict,
        inputs=[
            model_dropdown,
            race,
            gender,
            age,
            weight,
            payer_code,
            medical_specialty,
            admission_type_id,
            discharge_disposition_id,
            admission_source_id,
            time_in_hospital,
            num_lab_procedures,
            num_procedures,
            num_medications,
            number_outpatient,
            number_emergency,
            number_inpatient,
            number_diagnoses,
            diag_1,
            diag_2,
            diag_3,
            max_glu_serum,
            A1Cresult,
            metformin,
            repaglinide,
            nateglinide,
            chlorpropamide,
            glimepiride,
            acetohexamide,
            glipizide,
            glyburide,
            tolbutamide,
            pioglitazone,
            rosiglitazone,
            acarbose,
            miglitol,
            troglitazone,
            tolazamide,
            examide,
            citoglipton,
            insulin,
            glyburide_metformin,
            glipizide_metformin,
            glimepiride_pioglitazone,
            metformin_rosiglitazone,
            metformin_pioglitazone,
            change,
            diabetesMed
        ],
        outputs=[prediction_output]
    )

if __name__ == "__main__":
    Thread(target=run_metrics_server, daemon=True).start()
    logger.info("Prometheus metrics server started on port 9090")
    
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=8501,
        favicon_path="https://cdn-icons-png.flaticon.com/512/2966/2966327.png"
    )