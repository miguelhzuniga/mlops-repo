from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
from typing import List, Dict, Any

# Crear la aplicación FastAPI
app = FastAPI(title="Iris Species Predictor API", description="API para predecir especies de iris")

# Métricas Prometheus
PREDICTION_COUNT = Counter('api_predictions_total', 'Total de predicciones realizadas', ['species'])
PREDICTION_LATENCY = Histogram('api_prediction_latency_seconds', 'Tiempo de respuesta para predicciones')

# Rutas a los archivos del modelo
MODEL_PATH = "app/model.pkl"
MODEL_INFO_PATH = "app/model_info.pkl"

# Cargar el modelo y su información
model = joblib.load(MODEL_PATH)
print("Modelo KNN cargado correctamente")
request = {
  "sepal_length": 10,
  "sepal_width": 10,
  "petal_length": 10,
  "petal_width": 10
}
features = np.array([
    request["sepal_length"],
    request["sepal_width"],
    request["petal_length"],
    request["petal_width"]
]).reshape(1, -1)

model_info = joblib.load(MODEL_INFO_PATH)
feature_names = model_info.get('feature_names')

# Crear un DataFrame con los nombres correctos de las columnas si están disponibles
if feature_names:
    features_df = pd.DataFrame(features, columns=feature_names)
else:
    features_df = features
        
probabilities = model.predict_proba(features_df)[0]
print(probabilities)