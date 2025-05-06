from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import time
import os
import boto3
import joblib
import tempfile
import dill  # Añadido para la deserialización avanzada
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import logging
import uvicorn

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(title="API de Predicción de Diabetes",
              description="API para predicción de diabetes usando modelos de MLflow",
              version="1.0.0")

# Permitir CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Métricas de Prometheus
REQUESTS = Counter('diabetes_api_requests_total', 'Número total de solicitudes a la API')
PREDICTIONS = Counter('diabetes_api_predictions_total', 'Número total de predicciones realizadas')
PREDICTION_TIME = Histogram('diabetes_api_prediction_time_seconds', 
                           'Tiempo empleado en procesar solicitudes de predicción')
MODEL_ERRORS = Counter('diabetes_api_model_errors_total', 'Número total de errores del modelo')
PREPROCESSOR_ERRORS = Counter('diabetes_api_preprocessor_errors_total', 'Errores al cargar/aplicar el preprocesador')

# Configuración para MLflow con S3/MinIO
HOST_IP = "10.43.101.206"  # Mantenida la IP original
MLFLOW_S3_ENDPOINT_URL = f"http://{HOST_IP}:30382"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = "adminuser"
os.environ["AWS_SECRET_ACCESS_KEY"] = "securepassword123"

# URI del servidor de seguimiento MLflow
MLFLOW_TRACKING_URI = f"http://{HOST_IP}:30500"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Configuración del preprocesador - usando la misma ruta que en el DAG
BUCKET_NAME = "mlflow-artifacts"
PREPROCESSOR_KEY = "preprocessors/preprocessor.joblib"

# Caché del preprocesador y modelo para evitar cargas repetidas
preprocessor_cache = None
model_cache = None
model_version_cache = None
model_name_cache = None

# Columnas esperadas por el preprocesador - basado en el DAG de entrenamiento
# Estas columnas coinciden con las utilizadas durante el entrenamiento del modelo
ALL_EXPECTED_COLUMNS = [
    'encounter_id', 'patient_nbr', 'race', 'gender', 'age', 
    'weight', 'admission_type_id', 'discharge_disposition_id', 
    'admission_source_id', 'time_in_hospital', 'payer_code', 
    'medical_specialty', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses', 'diag_1', 'diag_2', 
    'diag_3', 'max_glu_serum', 'A1Cresult', 'a1cresult',
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone',
    'change', 'diabetesMed', 'diabetesmed'
]

# Extraemos las columnas numéricas y categóricas del DAG de entrenamiento
# para asegurar consistencia con el preprocesador almacenado en MinIO
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

# Mapeo de nombres con guiones bajos a nombres con guiones
FIELD_MAPPING = {
    'glyburide_metformin': 'glyburide-metformin',
    'glipizide_metformin': 'glipizide-metformin',
    'glimepiride_pioglitazone': 'glimepiride-pioglitazone',
    'metformin_rosiglitazone': 'metformin-rosiglitazone',
    'metformin_pioglitazone': 'metformin-pioglitazone'
}


class DiabetesFeatures(BaseModel):
    """Modelo Pydantic básico - solo campos esenciales"""
    race: str
    gender: str
    age: str
    admission_type_id: int
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    max_glu_serum: str
    A1Cresult: str
    insulin: str
    diabetesMed: str
    
    # Otros campos son opcionales
    class Config:
        extra = "allow"  # Permite campos adicionales no declarados


def normalize_field_names(data_dict):
    """
    Normaliza los nombres de los campos, reemplazando guiones bajos por guiones
    cuando sea necesario según el mapeo definido.
    """
    normalized = {}
    for key, value in data_dict.items():
        # Verificar si el campo está en el mapeo
        if key in FIELD_MAPPING:
            normalized_key = FIELD_MAPPING[key]
            logger.info(f"Normalizando campo: {key} -> {normalized_key}")
        else:
            normalized_key = key
        normalized[normalized_key] = value
    return normalized


def load_preprocessor():
    """Cargar el preprocesador desde MinIO - coincidiendo con la configuración del DAG de entrenamiento"""
    global preprocessor_cache
    
    if preprocessor_cache is not None:
        logger.info("Usando preprocesador en caché")
        return preprocessor_cache
    
    try:
        # Configurar cliente S3/MinIO igual que en el DAG de entrenamiento
        s3_client = boto3.client(
            's3',
            endpoint_url=MLFLOW_S3_ENDPOINT_URL,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
        )
        
        # Crear un archivo temporal para guardar el preprocesador
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_path = tmp_file.name
            
        # Descargar el preprocesador - misma ubicación que en el DAG
        logger.info(f"Descargando preprocesador desde s3://{BUCKET_NAME}/{PREPROCESSOR_KEY}")
        s3_client.download_file(BUCKET_NAME, PREPROCESSOR_KEY, temp_path)
        
        # Importar lo necesario para cargar el preprocesador
        import dill
        
        try:
            # Intentar cargar con joblib primero (método preferido según el DAG)
            preprocessor_cache = joblib.load(temp_path)
            logger.info("Preprocesador cargado exitosamente con joblib")
        except Exception as joblib_error:
            logger.warning(f"Error al cargar con joblib: {joblib_error}. Intentando con dill...")
            # Si falla, intentar con dill como respaldo
            with open(temp_path, 'rb') as f:
                preprocessor_cache = dill.load(f)
            logger.info("Preprocesador cargado exitosamente con dill")
        
        # Eliminar el archivo temporal
        os.unlink(temp_path)
        
        # Verificar que el preprocesador sea el esperado (ColumnTransformer)
        if hasattr(preprocessor_cache, 'transformers'):
            logger.info(f"Preprocesador verificado. Transformers: {len(preprocessor_cache.transformers)}")
        else:
            logger.warning("El preprocesador no parece ser un ColumnTransformer")
        
        return preprocessor_cache
    
    except Exception as e:
        PREPROCESSOR_ERRORS.inc()
        logger.error(f"Error al cargar el preprocesador: {str(e)}")
        raise


def get_production_model():
    """
    Obtener cualquier modelo que esté en estado de producción,
    sin buscar por un nombre específico.
    """
    global model_cache, model_version_cache, model_name_cache
    
    # Usar caché si está disponible
    if model_cache is not None and model_version_cache is not None and model_name_cache is not None:
        return model_cache, model_version_cache, model_name_cache
    
    try:
        # Obtener el cliente de MLflow
        client = mlflow.tracking.MlflowClient()
        
        # Método 1: Buscar modelos registrados y encontrar el que está en producción
        registered_models = client.search_registered_models()
        
        production_model = None
        model_name = None
        version = None
        
        # Recorrer todos los modelos registrados
        for reg_model in registered_models:
            model_name = reg_model.name
            logger.info(f"Verificando modelo: {model_name}")
            
            # Obtener detalles del modelo para ver sus versiones
            try:
                model_details = client.get_registered_model(model_name)
                
                # Buscar versiones en producción
                if hasattr(model_details, 'latest_versions'):
                    for model_version in model_details.latest_versions:
                        if hasattr(model_version, 'current_stage') and model_version.current_stage == "Production":
                            production_model = model_version
                            version = model_version.version
                            logger.info(f"Encontrado modelo en producción: {model_name} versión {version}")
                            break
                
                if production_model:
                    break
            except Exception as e:
                logger.warning(f"Error al obtener detalles del modelo {model_name}: {str(e)}")
        
        # Si no encontramos un modelo en producción, intentar otro enfoque
        if not production_model:
            logger.warning("No se encontró ningún modelo en estado 'Production', buscando todos los modelos")
            
            # Intentar obtener cualquier modelo, preferiblemente el más reciente
            if registered_models:
                model_name = registered_models[0].name
                versions = client.search_model_versions(f"name='{model_name}'")
                
                if versions:
                    # Ordenar por versión y tomar la más reciente
                    versions_sorted = sorted(
                        versions,
                        key=lambda x: int(x.version if hasattr(x, 'version') else 0),
                        reverse=True
                    )
                    
                    version = versions_sorted[0].version
                    logger.info(f"Usando la versión más reciente: {model_name} versión {version}")
                else:
                    raise ValueError(f"No se encontraron versiones para {model_name}")
            else:
                raise ValueError("No se encontraron modelos registrados")
        
        # Cargar el modelo
        model_uri = f"models:/{model_name}/{version}"
        logger.info(f"Cargando modelo desde: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Guardar en caché
        model_cache = model
        model_version_cache = version
        model_name_cache = model_name
        
        logger.info(f"Modelo cargado: {model_name} versión {version}")
        return model, version, model_name
    
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}")
        MODEL_ERRORS.inc()
        raise


def preprocess_input(input_data):
    """Preprocesar los datos de entrada con manejo estricto de tipos"""
    try:
        logger.info("Iniciando preprocesamiento de datos")
        logger.info(f"Columnas recibidas: {input_data.columns.tolist()}")
        
        # NUEVO: Convertir nombres de campo con guiones bajos a guiones
        for old_col, new_col in FIELD_MAPPING.items():
            if old_col in input_data.columns and new_col not in input_data.columns:
                input_data[new_col] = input_data[old_col]
                input_data = input_data.drop(old_col, axis=1)
                logger.info(f"Convertido campo {old_col} a {new_col}")
        
        # 1. Para columnas numéricas, asegurar que sean float o int
        for col in NUMERIC_COLUMNS:
            if col in input_data.columns:
                # Intentar convertir a numérico, coercionar errores a NaN
                try:
                    input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
                    # Reemplazar NaN con 0
                    input_data[col] = input_data[col].fillna(0).astype(float)
                except Exception as e:
                    logger.warning(f"Error al convertir columna {col} a numérico: {e}")
                    input_data[col] = 0.0
            else:
                # Si no existe la columna, añadirla con valores por defecto
                input_data[col] = 0.0
        
        # 2. Para columnas categóricas, asegurar que sean string
        for col in CATEGORICAL_COLUMNS:
            if col in input_data.columns:
                # Convertir a string y reemplazar valores nulos/vacíos
                input_data[col] = input_data[col].astype(str)
                input_data[col] = input_data[col].replace(['?', 'nan', 'None', 'NULL', 'NaN', ''], 'Unknown')
                
                # NUEVO: Manejo especial para valores con comparadores - CRÍTICO para evitar comparaciones str vs float
                if col in ['max_glu_serum', 'A1Cresult']:
                    # No intentar conversiones numéricas para estos campos, tratarlos siempre como categóricos
                    if input_data[col].str.contains('>').any() or input_data[col].str.contains('<').any():
                        logger.info(f"Detectados valores con comparadores en columna {col}")
            else:
                # Si no existe la columna, añadirla con valores por defecto
                if col in ['insulin', 'metformin', 'change'] or col.endswith('metformin') or '-' in col:
                    input_data[col] = 'No'  # Para medicamentos
                else:
                    input_data[col] = 'Unknown'  # Para otras categorías
        
        # 3. Manejar duplicados con diferente capitalización
        if 'A1Cresult' in input_data.columns and 'a1cresult' in ALL_EXPECTED_COLUMNS:
            input_data['a1cresult'] = input_data['A1Cresult']
        if 'diabetesMed' in input_data.columns and 'diabetesmed' in ALL_EXPECTED_COLUMNS:
            input_data['diabetesmed'] = input_data['diabetesMed']
        
        # 4. Eliminar la columna readmitted si está presente
        if "readmitted" in input_data.columns:
            input_data = input_data.drop("readmitted", axis=1)
        
        # Verificar antes de aplicar el preprocesador
        logger.info(f"Tipos de datos antes del preprocesamiento:")
        for col in input_data.columns:
            logger.info(f"  {col}: {input_data[col].dtype}")
        
        # Cargar y aplicar el preprocesador
        # El preprocesador debe ser el mismo que se utilizó en el entrenamiento (del DAG)
        preprocessor = load_preprocessor()
        
        # Prueba de conversión: Verificar que el DataFrame se puede convertir correctamente
        try:
            dummy_array = input_data.to_numpy()
            logger.info(f"Conversión a numpy exitosa: {dummy_array.shape}")
        except Exception as e:
            logger.error(f"Error al convertir DataFrame a numpy: {e}")
            # Intentar reparar cada columna individualmente
            for col in input_data.columns:
                try:
                    dummy = input_data[col].to_numpy()
                except Exception as col_e:
                    logger.error(f"Error en columna {col}: {col_e}")
                    if col in NUMERIC_COLUMNS:
                        input_data[col] = 0.0
                    else:
                        input_data[col] = 'Unknown'
        
        # Aquí se aplica el preprocesador - envuelto en un manejador de excepciones
        try:
            processed_data = preprocessor.transform(input_data)
            logger.info(f"Preprocesamiento exitoso: {processed_data.shape}")
            return processed_data
        except Exception as transform_error:
            logger.error(f"Error en transform: {transform_error}")
            # Intento de recuperación: usar solo columnas coincidentes con el DAG
            subset_columns = [col for col in input_data.columns if col in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
            input_subset = input_data[subset_columns].copy()
            
            # Verificar y volver a aplicar tipos
            for col in subset_columns:
                if col in NUMERIC_COLUMNS:
                    input_subset[col] = input_subset[col].astype(float)
                else:
                    input_subset[col] = input_subset[col].astype(str)
            
            # Intentar nuevamente con el subconjunto
            logger.info(f"Intentando con subconjunto de {len(subset_columns)} columnas")
            processed_data = preprocessor.transform(input_subset)
            return processed_data
    
    except Exception as e:
        logger.error(f"Error en el preprocesamiento: {str(e)}")
        logger.error(f"Columnas en los datos: {input_data.columns.tolist()}")
        PREPROCESSOR_ERRORS.inc()
        raise HTTPException(status_code=500, detail=f"Error en preprocesamiento: {str(e)}")


@app.get("/")
async def root():
    return {
        "mensaje": "API de Predicción de Diabetes",
        "documentación": "/docs",
        "métricas": "/metrics"
    }


@app.post("/predict")
async def predict(features: DiabetesFeatures):
    """Realizar una predicción utilizando el modelo"""
    REQUESTS.inc()
    
    with PREDICTION_TIME.time():
        try:
            # NUEVO: Normalizar nombres de campos antes de la conversión
            normalized_data = normalize_field_names(features.dict())
            
            # Convertir entrada a DataFrame
            input_data = pd.DataFrame([normalized_data])
            logger.info(f"Datos recibidos con {len(input_data.columns)} columnas")
            
            # Preprocesar los datos
            processed_data = preprocess_input(input_data)
            
            # Obtener el modelo
            model, model_version, model_name = get_production_model()
            
            # Realizar predicción
            prediction = model.predict(processed_data)
            PREDICTIONS.inc()
            
            # Interpretar la predicción
            prediction_value = prediction.tolist()[0]
            if isinstance(prediction_value, (np.ndarray, list)):
                prediction_value = prediction_value[0]
                       
            return {
                "prediccion": prediction_value
            }
        
        except Exception as e:
            logger.error(f"Error de predicción: {str(e)}")
            MODEL_ERRORS.inc()
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Endpoint de verificación de salud"""
    try:
        # Verificar que podemos cargar el preprocesador
        preprocessor = load_preprocessor()
        
        # Verificar que podemos cargar el modelo
        model, model_version, model_name = get_production_model()
        
        return {
            "estado": "saludable",
            "modelo_nombre": model_name,
            "modelo_version": model_version,
            "preprocesador_cargado": preprocessor is not None
        }
    except Exception as e:
        logger.error(f"Error en health check: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Endpoint para proporcionar métricas a Prometheus"""
    return generate_latest(), {"Content-Type": CONTENT_TYPE_LATEST}


@app.post("/test_preprocess")
async def test_preprocess(features: DiabetesFeatures):
    """Endpoint para probar solo el preprocesamiento (diagnóstico)"""
    try:
        # NUEVO: Normalizar nombres de campos antes de la conversión
        normalized_data = normalize_field_names(features.dict())
        
        # Convertir entrada a DataFrame
        input_data = pd.DataFrame([normalized_data])
        
        # Log de los tipos de datos
        dtypes_info = {col: str(input_data[col].dtype) for col in input_data.columns}
        
        # Preprocesar sin aplicar el modelo
        processed_data = preprocess_input(input_data)
        
        return {
            "mensaje": "Preprocesamiento exitoso",
            "columnas_entrada": input_data.columns.tolist(),
            "tipos_datos": dtypes_info,
            "forma_procesada": processed_data.shape
        }
    except Exception as e:
        logger.error(f"Error en test de preprocesamiento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Verificar configuración al inicio
    try:
        logger.info(f"Verificando conexión a MLflow en {MLFLOW_TRACKING_URI}")
        # Precargar para detectar errores temprano
        logger.info("Precargando preprocesador y modelo...")
        preprocessor = load_preprocessor()
        model, model_version, model_name = get_production_model()
        logger.info(f"Precarga completada: Modelo {model_name} v{model_version}")
    except Exception as e:
        logger.error(f"Error en la inicialización: {str(e)}")
    
    uvicorn.run("main_server:app", host="0.0.0.0", port=80, reload=True)
