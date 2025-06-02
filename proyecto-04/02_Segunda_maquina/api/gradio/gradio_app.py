# 📦 Importaciones principales
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
from fastapi import FastAPI, HTTPException, Response
from threading import Thread
from datetime import datetime
import psycopg2  # 📌 Para acceder a trainlogs.logs
import shap
import matplotlib.pyplot as plt

# 📊 Configuración de métricas Prometheus
REQUESTS = Counter('house_price_gradio_requests_total', 'Número total de solicitudes a la interfaz Gradio')
PREDICTIONS = Counter('house_price_gradio_predictions_total', 'Número total de predicciones realizadas')
PREDICTION_TIME = Histogram('house_price_gradio_prediction_time_seconds', 'Tiempo empleado en procesar solicitudes de predicción')
MODEL_ERRORS = Counter('house_price_gradio_model_errors_total', 'Errores del modelo')
PREPROCESSOR_ERRORS = Counter('house_price_gradio_preprocessor_errors_total', 'Errores al cargar o aplicar el preprocesador')
MODEL_LOADS = Counter('house_price_gradio_model_loads_total', 'Número de veces que se cargó un modelo')
REFRESH_CALLS = Counter('house_price_gradio_refresh_calls_total', 'Número de veces que se actualizó la lista de modelos')

# 🎛️ FastAPI para Prometheus y Logs
metrics_app = FastAPI(title="Prometheus Metrics for House Price Prediction")
@metrics_app.get("/")
async def root(): return {"message": "Prometheus Metrics Server for House Price Prediction"}


@metrics_app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type= CONTENT_TYPE_LATEST)

def run_metrics_server(): uvicorn.run(metrics_app, host="0.0.0.0", port=9090)

# 🌐 Configuración MLflow/MinIO
HOST_IP = "10.43.101.206"
os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{HOST_IP}:30382"
os.environ['AWS_ACCESS_KEY_ID'] = "adminuser"
os.environ['AWS_SECRET_ACCESS_KEY'] = "securepassword123"
mlflow.set_tracking_uri(f"http://{HOST_IP}:30500")
MINIO_ENDPOINT = f"http://{HOST_IP}:30382"
AWS_ACCESS_KEY = "adminuser"
AWS_SECRET_KEY = "securepassword123"
BUCKET_NAME = "mlflow-artifacts"
PREPROCESSOR_KEY = "preprocessors/preprocessor.joblib"
preprocessor_cache = None
loaded_models = {}
current_model_name = None
s3_client = boto3.client('s3', endpoint_url=MINIO_ENDPOINT, aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

NUMERIC_COLUMNS = ['bed', 'bath', 'acre_lot', 'house_size']
CATEGORICAL_COLUMNS = ['brokered_by', 'status', 'street', 'city', 'state', 'zip_code']

def get_logs(limit=100):
    try:
        conn = psycopg2.connect(
            host="10.43.101.175",
            port=5432,
            database="trainlogs",
            user="airflow",
            password="airflow"
        )
        cur = conn.cursor()
        cur.execute(f'SELECT * FROM logs ORDER BY id DESC LIMIT {limit};')
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        data = [dict(zip(columns, row)) for row in rows]
        conn.close()
        return data
    except Exception as e:
        return {"error": str(e)}

@metrics_app.get("/logs")
async def logs():
    data = get_logs()
    return {"logs": data}

# 🎛️ Preprocesador y funciones clave
def load_preprocessor():
    global preprocessor_cache
    if preprocessor_cache: return preprocessor_cache
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file: temp_path = tmp_file.name
    s3_client.download_file(BUCKET_NAME, PREPROCESSOR_KEY, temp_path)
    try: preprocessor_cache = joblib.load(temp_path)
    except: preprocessor_cache = dill.load(open(temp_path, 'rb'))
    os.unlink(temp_path)
    return preprocessor_cache

def preprocess_input(input_data):
    try:
        # Eliminar columnas no necesarias
        for col in ['id', 'price_per_sqft', 'price']:
            if col in input_data.columns:
                input_data = input_data.drop(columns=[col])
        input_data['prev_sold_year'] = input_data['prev_sold_date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").year if pd.notna(x) and x != '' else 0)
        input_data = input_data.drop(columns=['prev_sold_date'])
        for col in NUMERIC_COLUMNS: input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)
        for col in CATEGORICAL_COLUMNS + ['prev_sold_year']: input_data[col] = input_data[col].astype(str).fillna('Unknown')
        preprocessor = load_preprocessor()
        return preprocessor.transform(input_data)
    except Exception as e:
        PREPROCESSOR_ERRORS.inc()
        raise Exception(f"Error en el preprocesamiento: {e}")

def load_model(model_name):
    global current_model_name, loaded_models
    if model_name in loaded_models: current_model_name = model_name; return f"Modelo {model_name} ya cargado."
    client = mlflow.tracking.MlflowClient()
    model_uri = f"models:/{model_name}/Production"
    loaded_models[model_name] = mlflow.pyfunc.load_model(model_uri)
    current_model_name = model_name
    MODEL_LOADS.inc()
    return f"Modelo {model_name} cargado exitosamente."

def predict(model_name, brokered_by, status, bed, bath, acre_lot, street, city, state, zip_code, house_size, prev_sold_date):
    REQUESTS.inc()
    with PREDICTION_TIME.time():
        model_to_use = model_name if model_name else current_model_name
        if model_to_use not in loaded_models: return f"El modelo {model_to_use} no está cargado."
        input_dict = {'brokered_by': brokered_by, 'status': status, 'bed': bed, 'bath': bath, 'acre_lot': acre_lot,
                      'street': street, 'city': city, 'state': state, 'zip_code': zip_code, 'house_size': house_size,
                      'prev_sold_date': prev_sold_date}
        input_df = pd.DataFrame([input_dict])
        X_processed = preprocess_input(input_df)
        prediction = loaded_models[model_to_use].predict(X_processed)
        PREDICTIONS.inc()
        return f"<h3>El precio estimado de la casa es: <strong>${prediction[0]:,.2f} USD</strong></h3>"

def refresh_models():
    REFRESH_CALLS.inc()
    models = mlflow.tracking.MlflowClient().search_registered_models()
    choices = [model.name for model in models]
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None), f"{len(choices)} modelos disponibles."

def get_shap_summary_plot():
    try:
        model_to_use = current_model_name
        if model_to_use not in loaded_models:
            return "No hay modelo cargado."
        model = loaded_models[model_to_use]
        preprocessor = load_preprocessor()
        
        # Genera un dataset de prueba con datos sintéticos
        feature_names = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ['prev_sold_year']
        dummy_data = pd.DataFrame({
            'brokered_by': ['101640.0']*50,
            'status': ['for_sale']*50,
            'bed': [3]*50,
            'bath': [2]*50,
            'acre_lot': [0.25]*50,
            'street': ['1758218.0']*50,
            'city': ['East Windsor']*50,
            'state': ['Connecticut']*50,
            'zip_code': ['6016.0']*50,
            'house_size': [1500]*50,
            'prev_sold_date': ['2015-11-09']*50
        })
        X_processed = preprocess_input(dummy_data)
        
        explainer = shap.Explainer(model.predict, X_processed)
        shap_values = explainer(X_processed)
        
        fig, ax = plt.subplots(figsize=(10,6))
        shap.summary_plot(shap_values, features=X_processed, show=False)
        plt.tight_layout()
        return fig
    except Exception as e:
        return f"Error al generar SHAP: {e}"
      
# 📋 Gradio App actualizada
with gr.Blocks() as app:
    gr.Markdown("# 🏠 Predicción de Precios de Casas")
    
    # 🗂️ Pestañas
    with gr.Tabs():
        # 🔍 Pestaña de Predicción
        with gr.TabItem("Predicción"):
            model_dropdown = gr.Dropdown(label="Modelo a usar", choices=[])
            refresh_btn = gr.Button("Actualizar modelos")
            load_btn = gr.Button("Cargar modelo")
            load_output = gr.HTML()
            refresh_btn.click(refresh_models, outputs=[model_dropdown, load_output])
            load_btn.click(load_model, inputs=model_dropdown, outputs=load_output)
            gr.Markdown("### Ingrese los datos de la casa:")
            with gr.Row():
                brokered_by = gr.Textbox(label="Agencia/Corredor (ID o nombre)", value="101640.0")
                status = gr.Dropdown(label="Estado de la casa", choices=["for_sale", "ready_to_build"], value="for_sale")
                street = gr.Textbox(label="Calle (ID o nombre)", value="1758218.0")
                city = gr.Textbox(label="Ciudad", value="East Windsor")
                state = gr.Textbox(label="Estado", value="Connecticut")
                zip_code = gr.Textbox(label="Código Postal", value="6016.0")
            with gr.Row():
                bed = gr.Number(label="Habitaciones", value=3)
                bath = gr.Number(label="Baños", value=2)
                acre_lot = gr.Number(label="Tamaño del terreno (acres)", value=0.25)
                house_size = gr.Number(label="Área de la casa (sqft)", value=1500)
                prev_sold_date = gr.Textbox(label="Fecha de venta anterior (YYYY-MM-DD)", value="2015-11-09")
            predict_btn = gr.Button("Predecir precio")
            pred_output = gr.HTML()
            predict_btn.click(predict, inputs=[model_dropdown, brokered_by, status, bed, bath, acre_lot, street, city, state, zip_code, house_size, prev_sold_date], outputs=pred_output)
            
    # Reemplaza toda la pestaña de Logs con este código:

        # 📊 Pestaña de Logs
        with gr.TabItem("Logs (trainlogs.logs)"):
            gr.Markdown("### Últimos registros de la tabla `trainlogs.logs`")
            fetch_logs_btn = gr.Button("Actualizar Logs")
            
            # ✅ CORREGIDO: Sin headers vacíos
            logs_table = gr.Dataframe()
            
            def fetch_logs_gradio():
                data = get_logs()
                if isinstance(data, dict) and "error" in data:
                    return pd.DataFrame([{"Error": data["error"]}])
                if not data:
                    return pd.DataFrame([{"Mensaje": "No hay registros disponibles"}])
                
                # Convertir la lista de diccionarios directamente a DataFrame
                df = pd.DataFrame(data)
                return df
            
            fetch_logs_btn.click(fetch_logs_gradio, outputs=logs_table)


        # 🌟 Pestaña de Análisis SHAP
        with gr.TabItem("Análisis SHAP"):
            gr.Markdown("### Importancia de características (SHAP Summary Plot)")
            shap_btn = gr.Button("Generar Análisis SHAP")
            shap_plot = gr.Plot()
            shap_btn.click(get_shap_summary_plot, outputs=shap_plot)

# 🚀 Main
if __name__ == "__main__":
    Thread(target=run_metrics_server, daemon=True).start()
    app.launch(server_name="0.0.0.0", server_port=8501)
