from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import os
import boto3
import tempfile
import joblib
import dill
import uvicorn
from datetime import datetime

app = FastAPI(
    title="API de Predicción de Precios de Casas",
    description="Predice el precio de una casa usando MLflow y FastAPI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUESTS = Counter('house_api_requests_total', 'Número total de solicitudes')
PREDICTIONS = Counter('house_api_predictions_total', 'Número total de predicciones')
PREDICTION_TIME = Histogram('house_api_prediction_time_seconds', 'Tiempo de predicción')
MODEL_ERRORS = Counter('house_api_model_errors_total', 'Errores del modelo')

# Configuración MLflow/MinIO
HOST_IP = "10.43.101.206"
MLFLOW_S3_ENDPOINT_URL = f"http://{HOST_IP}:30382"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = "adminuser"
os.environ["AWS_SECRET_ACCESS_KEY"] = "securepassword123"
MLFLOW_TRACKING_URI = f"http://{HOST_IP}:30500"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
BUCKET_NAME = "mlflow-artifacts"
PREPROCESSOR_KEY = "preprocessors/preprocessor.joblib"

preprocessor_cache = None
model_cache = None

NUMERIC_COLUMNS = ['bed', 'bath', 'acre_lot', 'house_size']
CATEGORICAL_COLUMNS = ['brokered_by', 'status', 'street', 'city', 'state', 'zip_code']

class HouseFeatures(BaseModel):
    brokered_by: str
    status: str
    bed: float
    bath: float
    acre_lot: float
    street: str
    city: str
    state: str
    zip_code: str
    house_size: float
    prev_sold_date: str

def load_preprocessor():
    global preprocessor_cache
    if preprocessor_cache:
        return preprocessor_cache
    s3_client = boto3.client(
        's3',
        endpoint_url=MLFLOW_S3_ENDPOINT_URL,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        temp_path = tmp_file.name
    s3_client.download_file(BUCKET_NAME, PREPROCESSOR_KEY, temp_path)
    try:
        preprocessor_cache = joblib.load(temp_path)
    except:
        with open(temp_path, 'rb') as f:
            preprocessor_cache = dill.load(f)
    os.unlink(temp_path)
    return preprocessor_cache

def preprocess_input(input_data):
    try:
        input_data['prev_sold_year'] = input_data['prev_sold_date'].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d").year if pd.notna(x) else 0
        )
        input_data = input_data.drop(columns=['prev_sold_date'])
        for col in NUMERIC_COLUMNS:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)
        for col in CATEGORICAL_COLUMNS + ['prev_sold_year']:
            input_data[col] = input_data[col].astype(str).fillna('Unknown')
        preprocessor = load_preprocessor()
        return preprocessor.transform(input_data)
    except Exception as e:
        MODEL_ERRORS.inc()
        raise HTTPException(status_code=500, detail=f"Error en preprocesamiento: {str(e)}")

def get_model():
    global model_cache
    if model_cache:
        return model_cache
    client = mlflow.tracking.MlflowClient()
    models = client.search_registered_models()
    for m in models:
        for v in client.search_model_versions(f"name='{m.name}'"):
            if v.current_stage == "Production":
                model_cache = mlflow.pyfunc.load_model(f"models:/{m.name}/Production")
                return model_cache
    raise Exception("No hay modelos en producción en MLflow.")

@app.get("/")
async def root():
    return {"message": "API para predicción de precios de casas", "endpoints": ["/predict", "/metrics", "/health"]}

@app.post("/predict")
async def predict(features: HouseFeatures):
    REQUESTS.inc()
    with PREDICTION_TIME.time():
        try:
            input_dict = features.dict()
            input_df = pd.DataFrame([input_dict])
            processed = preprocess_input(input_df)
            model = get_model()
            prediction = model.predict(processed)
            PREDICTIONS.inc()
            return {"prediccion_precio_usd": float(prediction[0])}
        except Exception as e:
            MODEL_ERRORS.inc()
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return generate_latest(), {"Content-Type": CONTENT_TYPE_LATEST}

@app.get("/health")
async def health():
    try:
        preprocessor = load_preprocessor()
        model = get_model()
        return {"status": "ok", "preprocessor_loaded": True, "model_loaded": True}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/test_preprocess")
async def test_preprocess(features: HouseFeatures):
    try:
        input_dict = features.dict()
        input_df = pd.DataFrame([input_dict])
        processed = preprocess_input(input_df)
        return {"mensaje": "Preprocesamiento exitoso", "forma_procesada": processed.shape}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main_server:app", host="0.0.0.0", port=80, reload=True)
