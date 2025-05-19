from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

# Crear la aplicación FastAPI
app = FastAPI(title="Iris Species Predictor API", description="API para predecir especies de iris")

# Métricas Prometheus
PREDICTION_COUNT = Counter('api_predictions_total', 'Total de predicciones realizadas', ['species'])
PREDICTION_LATENCY = Histogram('api_prediction_latency_seconds', 'Tiempo de respuesta para predicciones')

# Rutas a los archivos del modelo
MODEL_PATH = "model.pkl"
MODEL_INFO_PATH = "model_info.pkl"

# Cargar el modelo y su información
try:
    model = joblib.load(MODEL_PATH)
    print("Modelo KNN cargado correctamente")
    
    # Intentar cargar la información del modelo
    try:
        model_info = joblib.load(MODEL_INFO_PATH)
        feature_names = model_info.get('feature_names')
        class_names = model_info.get('class_names')
        print(f"Clases del modelo: {class_names}")
    except:
        print("Información del modelo no disponible, usando valores predeterminados")
        feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        class_names = ['setosa', 'versicolor', 'virginica']
        
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None
    feature_names = []
    class_names = []

# Definir el esquema de entrada
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Definir el esquema de salida
class IrisResponse(BaseModel):
    species: str
    probability: float
    processing_time: float

@app.get("/")
def root():
    return {"message": "Iris Species Predictor API", "endpoints": ["/predict", "/metrics"]}

@app.post("/predict", response_model=IrisResponse)
def predict(request: IrisRequest):
    # Verificar si el modelo está cargado
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")
    
    # Medir el tiempo de procesamiento
    start_time = time.time()
    
    try:
        # Preprocesar los datos
        features = np.array([
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width
        ]).reshape(1, -1)
        
        # Crear un DataFrame con los nombres correctos de las columnas si están disponibles
        if feature_names:
            features_df = pd.DataFrame(features, columns=feature_names)
        else:
            features_df = features
        
        # Realizar predicción
        species_idx_or_name = model.predict(features_df)[0]
        
        # Determinar el nombre de la especie
        if isinstance(species_idx_or_name, (int, np.integer)):
            species = class_names[species_idx_or_name] if class_names else str(species_idx_or_name)
        else:
            species = str(species_idx_or_name)
        
        # Obtener probabilidades si el modelo lo soporta
        try:
            probabilities = model.predict_proba(features_df)[0]
            probability = float(max(probabilities))
        except:
            probability = 1.0  # Si el modelo no soporta probabilidades
        
        # Registrar métricas
        PREDICTION_COUNT.labels(species=species).inc()
        
        # Calcular tiempo de procesamiento
        processing_time = time.time() - start_time
        PREDICTION_LATENCY.observe(processing_time)
        
        return {
            "species": species,
            "probability": probability,
            "processing_time": processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Endpoint para proporcionar métricas a Prometheus"""
    return generate_latest(), {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)