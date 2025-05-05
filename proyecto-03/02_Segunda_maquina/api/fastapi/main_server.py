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

# Configuración para MLflow con S3/MinIO
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://10.43.101.206:30382"
os.environ["AWS_ACCESS_KEY_ID"] = "adminuser"
os.environ["AWS_SECRET_ACCESS_KEY"] = "securepassword123"

# URI del servidor de seguimiento MLflow (usando HOST_IP)
MLFLOW_TRACKING_URI = f"http://10.43.101.206:30500"  # Ajusta el puerto según tu configuración
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Nombre del modelo registrado
MODEL_NAME = "diabetes-model"

class DiabetesFeatures(BaseModel):
    """Modelo Pydantic para los datos de entrada"""
    # Características basadas en el conjunto de datos de diabetes procesado
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
    # Campos opcionales que pueden no ser requeridos para la predicción
    readmitted: Optional[str] = None


def get_production_model():
    """
    Obtener el modelo de producción más reciente de MLflow.
    Siempre busca el modelo sin importar si ya se encontró antes,
    para asegurarse de tener la versión más reciente.
    """
    try:
        # Obtener el modelo de producción de MLflow
        client = mlflow.tracking.MlflowClient()
        
        # Obtener el modelo registrado
        registered_model = client.get_registered_model(MODEL_NAME)
        
        # Encontrar la última versión en producción
        production_versions = [mv for mv in registered_model.latest_versions 
                               if mv.current_stage == "Production"]
        
        if not production_versions:
            raise ValueError("No se encontró ningún modelo en producción")
        
        # Ordenar por versión en caso de que haya múltiples modelos en producción
        # y tomar el más reciente
        production_versions.sort(key=lambda x: int(x.version), reverse=True)
        latest_production = production_versions[0]
        
        model_version = latest_production.version
        model_name = latest_production.name
        
        # Cargar el modelo
        model_uri = f"models:/{model_name}/{model_version}"
        logger.info(f"Cargando modelo desde: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        
        logger.info(f"Modelo de producción cargado: {model_name} versión {model_version}")
        return model, model_version, model_name
    
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}")
        MODEL_ERRORS.inc()
        raise


@app.get("/")
async def root():
    return {
        "mensaje": "API de Predicción de Diabetes",
        "documentación": "/docs",
        "métricas": "/metrics"
    }


@app.post("/predict")
async def predict(features: DiabetesFeatures):
    """Realizar una predicción utilizando el último modelo en producción"""
    REQUESTS.inc()
    
    with PREDICTION_TIME.time():
        try:
            # Obtener el modelo de producción (siempre busca el más reciente)
            model, model_version, model_name = get_production_model()
            
            # Convertir entrada a DataFrame
            input_data = pd.DataFrame([features.dict()])
            
            # Eliminar la columna readmitted si está presente (variable objetivo)
            if "readmitted" in input_data.columns:
                input_data = input_data.drop("readmitted", axis=1)
            
            # Realizar transformaciones adicionales basadas en el procesamiento de datos
            # Estas son similares a las que se aplicaron en el DAG process_data
            
            # Gestionar valores nulos o inválidos
            input_data = input_data.replace(['?', '', 'None', 'NULL'], np.nan)
            
            # Reemplazar valores nulos en columnas categóricas
            if 'race' in input_data.columns:
                input_data['race'] = input_data['race'].fillna('Unknown')
            if 'gender' in input_data.columns:
                input_data['gender'] = input_data['gender'].fillna('Unknown')
            if 'age' in input_data.columns:
                input_data['age'] = input_data['age'].fillna('Unknown')
            
            # Asegurar que las columnas numéricas sean numéricas
            numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                            'num_medications', 'number_outpatient', 'number_emergency', 
                            'number_inpatient', 'number_diagnoses']
            
            for col in numeric_cols:
                if col in input_data.columns:
                    input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
                    input_data[col] = input_data[col].fillna(0)
            
            # Gestionar columnas categóricas (codificación one-hot)
            categorical_cols_to_encode = ['gender', 'age', 'max_glu_serum', 'A1Cresult', 'diabetesMed']
            
            for col in categorical_cols_to_encode:
                if col in input_data.columns:
                    # Crear columnas dummy con el prefijo adecuado
                    dummies = pd.get_dummies(input_data[col], prefix=col, drop_first=True)
                    for dummy_col in dummies.columns:
                        input_data[dummy_col] = dummies[dummy_col].astype(int)
            
            # Derivar características adicionales
            if 'insulin' in input_data.columns:
                input_data['insulin_used'] = input_data['insulin'].apply(lambda x: 0 if x == 'No' else 1)
            
            logger.info(f"Datos de entrada procesados: {input_data.columns.tolist()}")
            
            # Realizar predicción
            prediction = model.predict(input_data)
            PREDICTIONS.inc()
            
            # Interpretar la predicción - ajustar según el modelo específico
            prediction_value = prediction.tolist()[0]
            
            # Manejar diferentes tipos de modelos y salidas
            if isinstance(prediction_value, (np.ndarray, list)):
                prediction_value = prediction_value[0]
            
            # En caso de que el modelo devuelva probabilidades
            prediction_label = "NO" if prediction_value < 0.5 else "YES"
            
            return {
                "prediccion": float(prediction_value),
                "etiqueta": prediction_label,
                "modelo_nombre": model_name,
                "modelo_version": model_version
            }
        
        except Exception as e:
            logger.error(f"Error de predicción: {str(e)}")
            MODEL_ERRORS.inc()
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Endpoint de verificación de salud"""
    try:
        # Intentar obtener el modelo para verificar si la conexión MLflow está funcionando
        model, model_version, model_name = get_production_model()
        return {
            "estado": "saludable",
            "modelo_nombre": model_name,
            "modelo_version": model_version
        }
    except Exception as e:
        logger.error(f"Error en health check: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Endpoint para proporcionar métricas a Prometheus"""
    return generate_latest(), {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    # Intenta conectar con MLflow al inicio para verificar la configuración
    try:
        client = mlflow.tracking.MlflowClient()
        logger.info(f"Conexión exitosa a MLflow en {MLFLOW_TRACKING_URI}")
        models = client.search_registered_models()
        logger.info(f"Modelos registrados: {[m.name for m in models]}")
    except Exception as e:
        logger.error(f"Error al conectar con MLflow: {str(e)}")
    
    uvicorn.run("main_server:app", host="0.0.0.0", port=80, reload=True)