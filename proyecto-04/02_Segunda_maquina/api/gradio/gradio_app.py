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
            database="airflow",
            user="airflow",
            password="airflow"
        )
        cur = conn.cursor()
        cur.execute(f'SELECT * FROM trainlogs.logs ORDER BY id DESC LIMIT {limit};')
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
    """SHAP analysis ultra-optimizado para velocidad"""
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
    
    try:
        # ✅ Verificaciones básicas
        if current_model_name not in loaded_models:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "❌ No hay modelo cargado", ha='center', va='center', 
                   fontsize=16, transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
            ax.axis('off')
            return fig
        
        try:
            import shap
            print("✅ SHAP importado")
        except ImportError:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "❌ SHAP no disponible", ha='center', va='center', 
                   fontsize=16, transform=ax.transAxes)
            ax.axis('off')
            return fig
        
        model = loaded_models[current_model_name]
        print(f"🚀 Iniciando SHAP ultra-rápido para: {current_model_name}")
        
        # 🚀 OPTIMIZACIÓN 1: Dataset MÍNIMO (solo 3 muestras)
        shap_samples = pd.DataFrame({
            'bed': [2, 3, 4],          # Solo 3 valores
            'bath': [1, 2, 3], 
            'acre_lot': [0.2, 0.3, 0.4],
            'house_size': [1200, 1500, 2000],
            'prev_sold_year': [2020, 2019, 2018]
        })
        
        print(f"🚀 Dataset mínimo: {shap_samples.shape}")
        
        # 🚀 OPTIMIZACIÓN 2: Función wrapper ultra-simple
        def fast_predict(X_numeric):
            """Predicción optimizada para velocidad"""
            batch_predictions = []
            
            # Procesar en batch para eficiencia
            records = []
            for i in range(X_numeric.shape[0]):
                records.append({
                    'bed': int(X_numeric[i, 0]),
                    'bath': int(X_numeric[i, 1]),
                    'acre_lot': float(X_numeric[i, 2]), 
                    'house_size': int(X_numeric[i, 3]),
                    'prev_sold_year': int(X_numeric[i, 4]),
                    # Valores fijos para velocidad
                    'brokered_by': '101640.0',
                    'status': 'for_sale', 
                    'street': '1758218.0',
                    'city': 'East Windsor',
                    'state': 'Connecticut',
                    'zip_code': '6016.0',
                    'prev_sold_date': '2020-01-01'  # Fijo para velocidad
                })
            
            # Procesar todo el batch de una vez
            df_batch = pd.DataFrame(records)
            processed_batch = preprocess_input(df_batch)
            predictions = model.predict(processed_batch)
            
            return predictions
        
        # 🚀 OPTIMIZACIÓN 3: Background ULTRA-MÍNIMO (solo 1 muestra)
        X_numeric = shap_samples[['bed', 'bath', 'acre_lot', 'house_size', 'prev_sold_year']].values
        background = X_numeric[:1]  # Solo 1 muestra de background
        
        print("🚀 Creando explainer con background mínimo...")
        
        # 🚀 OPTIMIZACIÓN 4: Usar TreeExplainer si es LightGBM (más rápido)
        try:
            # Intentar acceder al modelo original de LightGBM
            if hasattr(model, '_model_impl') or hasattr(model, 'predict'):
                # Para modelos MLflow, intentar TreeExplainer que es más rápido
                explainer = shap.Explainer(fast_predict, background, max_evals=50)  # Limitar evaluaciones
            else:
                explainer = shap.Explainer(fast_predict, background)
        except:
            explainer = shap.Explainer(fast_predict, background)
        
        # 🚀 OPTIMIZACIÓN 5: Analizar solo 2 muestras
        X_analysis = X_numeric[:2]  # Solo 2 muestras
        
        print("🚀 Calculando SHAP values (optimizado)...")
        
        # 🚀 OPTIMIZACIÓN 6: Agregar timeout y parámetros de velocidad
        import time
        start_time = time.time()
        
        try:
            # Usar parámetros para acelerar el cálculo
            shap_values = explainer(X_analysis, max_evals=100, silent=True)
        except TypeError:
            # Si no soporta parámetros adicionales
            shap_values = explainer(X_analysis)
        
        calc_time = time.time() - start_time
        print(f"✅ SHAP calculado en {calc_time:.1f} segundos: {shap_values.values.shape}")
        
        # 🚀 CREAR PLOT RÁPIDO
        fig, ax = plt.subplots(figsize=(10, 6))
        
        feature_names = ['Habitaciones', 'Baños', 'Terreno', 'Tamaño casa', 'Año venta']
        
        # Plot simplificado para velocidad
        shap.summary_plot(
            shap_values.values, 
            X_analysis,
            feature_names=feature_names,
            show=False,
            max_display=5,
            plot_size=(10, 6)
        )
        
        plt.title(f"SHAP Rápido - {current_model_name}\n(Análisis de {X_analysis.shape[0]} muestras en {calc_time:.1f}s)", 
                 fontsize=12)
        plt.tight_layout()
        
        print(f"✅ Plot completado en {time.time() - start_time:.1f}s total")
        return fig
        
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        error_text = f"❌ Error SHAP:\n\n{str(e)[:200]}..."
        if "timeout" in str(e).lower() or "slow" in str(e).lower():
            error_text += "\n\n🚀 El modelo es muy complejo para SHAP.\nIntenta la versión simplificada."
        
        ax.text(0.5, 0.5, error_text, ha='center', va='center', fontsize=10, 
               transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral"))
        ax.axis('off')
        plt.title("SHAP - Error de velocidad", fontsize=14)
        
        print(f"❌ Error SHAP: {str(e)}")
        return fig
        
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
