import os
import mlflow
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge
import time
import logging
from app.logic.model import load_production_model
from app.logic.preprocessing import preprocess_data
import uvicorn


logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="API de Predicción para Diabetes",
             description="API para predecir la readmisión hospitalaria de pacientes con diabetes",
             version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

PREDICTION_COUNT = Counter('prediction_count', 'Contador de predicciones realizadas')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Latencia de las solicitudes de predicción')
MODEL_VERSION = Gauge('model_version', 'Versión actual del modelo en uso')
PREDICTION_DISTRIBUTION = Counter('prediction_distribution', 'Distribución de las clases de predicción', ['prediction'])

class DiabetesData(BaseModel):
    gender: str
    age: str
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
    diabetesMed: str
    
    diag_1: Optional[str] = ""
    diag_2: Optional[str] = ""
    diag_3: Optional[str] = ""
    
    metformin: Optional[str] = ""
    repaglinide: Optional[str] = ""
    nateglinide: Optional[str] = ""
    chlorpropamide: Optional[str] = ""
    glimepiride: Optional[str] = ""
    acetohexamide: Optional[str] = ""
    glipizide: Optional[str] = ""
    glyburide: Optional[str] = ""
    tolbutamide: Optional[str] = ""
    pioglitazone: Optional[str] = ""
    rosiglitazone: Optional[str] = ""
    acarbose: Optional[str] = ""
    miglitol: Optional[str] = ""
    troglitazone: Optional[str] = ""
    tolazamide: Optional[str] = ""
    examide: Optional[str] = ""
    citoglipton: Optional[str] = ""
    insulin: Optional[str] = ""
    
    class Config:
        schema_extra = {
            "example": {
                "gender": "Male",
                "age": "[50-60)",
                "time_in_hospital": 7,
                "num_lab_procedures": 45,
                "num_procedures": 1,
                "num_medications": 18,
                "number_outpatient": 0,
                "number_emergency": 0,
                "number_inpatient": 1,
                "number_diagnoses": 9,
                "max_glu_serum": "Norm",
                "A1Cresult": "None",
                "diabetesMed": "Yes",
                "diag_1": "428",
                "diag_2": "428",
                "diag_3": "250",
                "insulin": "Steady"
            }
        }

class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    run_id: str
    creation_timestamp: str
    description: Optional[str] = None

@app.get("/", status_code=200)
async def root():
    """Endpoint principal para verificar que la API está funcionando"""
    return {"message": "API de Predicción de Diabetes está funcionando. Ir a /docs para la documentación."}

@app.get("/health", status_code=200)
async def health_check():
    """Endpoint para verificar la salud de la API"""
    return {"status": "healthy"}

@app.get("/model_info", response_model=ModelInfo)
async def get_model_info():
    """Obtener información sobre el modelo cargado actualmente"""
    try:
        _, model_info = load_production_model(MLFLOW_TRACKING_URI)
        return model_info
    except Exception as e:
        logger.error(f"Error al obtener información del modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(data: DiabetesData):
    """
    Realizar una predicción de readmisión hospitalaria
    Retorna: Clase de predicción y probabilidad
    """
    start_time = time.time()
    
    try:
        model, model_info = load_production_model(MLFLOW_TRACKING_URI)
        
        processed_data = preprocess_data(data)
        
        prediction = model.predict(processed_data)
        
        predicted_class = prediction[0]
        
        PREDICTION_COUNT.inc()
        prediction_latency = time.time() - start_time
        PREDICTION_LATENCY.observe(prediction_latency)
        PREDICTION_DISTRIBUTION.labels(prediction=str(predicted_class)).inc()
        
        result = {
            "prediction": str(predicted_class),
            "readmission_status": "Readmitido" if predicted_class == 1 else "No Readmitido",
            "model_info": model_info,
            "processing_time_ms": round(prediction_latency * 1000, 2)
        }
        
        logger.info(f"Predicción completada: {result['readmission_status']}")
        return result
        
    except Exception as e:
        logger.error(f"Error en la predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware para agregar el tiempo de procesamiento al encabezado de respuesta"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)