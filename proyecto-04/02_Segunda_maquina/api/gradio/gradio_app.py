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
import psycopg2
import pandas as pd

def predict(model_name, brokered_by, status, bed, bath, acre_lot, street, city, state, zip_code, house_size, prev_sold_date):
    REQUESTS.inc()
    with PREDICTION_TIME.time():
        model_to_use = model_name if model_name else current_model_name
        if model_to_use not in loaded_models:
            return f"El modelo {model_to_use} no está cargado."
        
        # Crear input_dict
        input_dict = {
            'brokered_by': brokered_by,
            'status': status,
            'bed': bed,
            'bath': bath,
            'acre_lot': acre_lot,
            'street': street,
            'city': city,
            'state': state,
            'zip_code': zip_code,
            'house_size': house_size,
            'prev_sold_date': prev_sold_date
        }

        # Preprocesar datos
        input_df = pd.DataFrame([input_dict])
        X_processed = preprocess_input(input_df)

        # Hacer predicción
        prediction = loaded_models[model_to_use].predict(X_processed)[0]

        # Calcular price_per_sqft
        price_per_sqft = prediction / house_size if house_size else None

        # Conectar a la base de datos PostgreSQL
        try:
            conn = psycopg2.connect(
                host="10.43.101.175",
                port=5432,
                database="airflow",
                user="airflow",
                password="airflow"
            )
            cursor = conn.cursor()

            # Insertar en rawdata.houses
            insert_query = """
                INSERT INTO rawdata.houses 
                (brokered_by, status, bed, bath, acre_lot, street, city, state, zip_code, house_size, prev_sold_date, data_origin, price)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                brokered_by,
                status,
                bed,
                bath,
                acre_lot,
                street,
                city,
                state,
                zip_code,
                house_size,
                prev_sold_date,
                'user',  # data_origin
                prediction
            )

            cursor.execute(insert_query, values)
            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            return f"Error al insertar en la base de datos: {str(e)}"

        PREDICTIONS.inc()
        return f"<h3>El precio estimado de la casa es: <strong>${prediction:,.2f} USD</strong></h3>"

def refresh_models():
    REFRESH_CALLS.inc()
    client = mlflow.tracking.MlflowClient()

    # Buscar modelos registrados
    models = client.search_registered_models()

    # Filtrar modelos con alguna versión en Production
    production_models = []
    for model in models:
        # Buscar versiones de este modelo
        for version in model.latest_versions:
            if version.current_stage == 'Production':
                production_models.append(model.name)
                break  # Si al menos una versión está en Production, incluimos el modelo

    # Eliminar duplicados si hay (por seguridad)
    production_models = list(set(production_models))

    return gr.Dropdown(
        choices=production_models, 
        value=production_models[0] if production_models else None
    ), f"{len(production_models)} modelos en Production disponibles."

# ✅ Actualización: soporte para DecisionTreeRegressor y LightGBM
def get_shap_summary_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()

    try:
        if current_model_name not in loaded_models:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "❌ No hay modelo cargado", ha='center', va='center', 
                   fontsize=16, transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
            ax.axis('off')
            return fig

        import shap
        print("✅ SHAP importado")

        model = loaded_models[current_model_name]
        print(f"🎯 Iniciando SHAP para: {current_model_name}")

        # Obtener nombres de características
        def get_feature_names_from_preprocessor():
            try:
                preprocessor = load_preprocessor()
                if hasattr(preprocessor, 'get_feature_names_out'):
                    return list(preprocessor.get_feature_names_out())
                elif hasattr(preprocessor, 'get_feature_names'):
                    return list(preprocessor.get_feature_names())
                else:
                    return get_feature_names_manual(preprocessor)
            except Exception as e:
                print(f"⚠️ Error obteniendo nombres del preprocesador: {e}")
                return None

        def get_feature_names_manual(preprocessor):
            try:
                feature_names = []
                if hasattr(preprocessor, 'transformers_'):
                    for name, transformer, features in preprocessor.transformers_:
                        if name == 'num':
                            feature_names.extend([f"num__{feat}" for feat in features])
                        elif name == 'cat':
                            if hasattr(transformer, 'categories_'):
                                for i, feature in enumerate(features):
                                    categories = transformer.categories_[i]
                                    for category in categories:
                                        feature_names.append(f"cat__{feature}__{category}")
                return feature_names
            except Exception as e:
                print(f"⚠️ Error en extracción manual: {e}")
                return None

        def create_feature_mapping(feature_names, shap_values):
            important_patterns = {
                'bed': 'Habitaciones',
                'bath': 'Baños',
                'acre_lot': 'Terreno (acres)',
                'house_size': 'Tamaño casa (sqft)',
                'prev_sold_year': 'Año venta anterior',
                'status__for_sale': 'Estado: En venta',
                'status__ready_to_build': 'Estado: Listo construir',
                'state__Connecticut': 'Estado: Connecticut',
                'state__New York': 'Estado: Nueva York',
                'city__East Windsor': 'Ciudad: East Windsor',
                'brokered_by': 'Agencia inmobiliaria'
            }

            feature_importance = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(feature_importance)[-20:]

            mapped_names = []
            for idx in top_indices:
                original_name = feature_names[idx] if feature_names and idx < len(feature_names) else f"Feature_{idx}"
                descriptive_name = original_name
                for pattern, desc_name in important_patterns.items():
                    if pattern in original_name:
                        descriptive_name = desc_name
                        break
                if descriptive_name == original_name and feature_names:
                    if 'num__' in original_name:
                        descriptive_name = original_name.replace('num__', '').replace('_', ' ').title()
                    elif 'cat__' in original_name:
                        parts = original_name.replace('cat__', '').split('__')
                        descriptive_name = f"{parts[0].title()}: {parts[1]}" if len(parts) >= 2 else parts[0].title()
                mapped_names.append((idx, descriptive_name, feature_importance[idx]))
            mapped_names.sort(key=lambda x: x[2], reverse=True)
            return mapped_names

        # Crear datos de muestra
        shap_samples = pd.DataFrame({
            'bed': [2, 3, 4, 5, 2],
            'bath': [1, 2, 3, 3, 1],
            'acre_lot': [0.15, 0.25, 0.35, 0.45, 0.20],
            'house_size': [1000, 1500, 2000, 2500, 1200],
            'prev_sold_year': [2020, 2019, 2018, 2017, 2021]
        })

        full_samples = []
        for _, row in shap_samples.iterrows():
            full_record = {
                'bed': row['bed'], 'bath': row['bath'], 'acre_lot': row['acre_lot'],
                'house_size': row['house_size'], 'prev_sold_year': int(row['prev_sold_year']),
                'brokered_by': '101640.0', 'status': 'for_sale', 'street': '1758218.0',
                'city': 'East Windsor', 'state': 'Connecticut', 'zip_code': '6016.0',
                'prev_sold_date': f"{int(row['prev_sold_year'])}-01-01"
            }
            full_samples.append(full_record)

        full_df = pd.DataFrame(full_samples)
        X_processed = preprocess_input(full_df)
        if hasattr(X_processed, 'toarray'):
            X_processed = X_processed.toarray()
        print(f"✅ Datos preprocesados: {X_processed.shape}")

        feature_names = get_feature_names_from_preprocessor()
        if feature_names:
            print(f"✅ {len(feature_names)} nombres de características obtenidos")
        else:
            print("⚠️ No se pudieron obtener nombres, usando genéricos")

        # SHAP explainer universal
        print("🚀 Creando TreeExplainer universal...")
        explainer = shap.TreeExplainer(model)

        print("🚀 Calculando SHAP values...")
        X_analysis = X_processed[:3]
        shap_values = explainer.shap_values(X_analysis)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        print(f"✅ SHAP values calculados: {shap_values.shape}")

        mapped_features = create_feature_mapping(feature_names, shap_values)
        top_features = mapped_features[:15]
        top_indices = [item[0] for item in top_features]
        top_names = [item[1] for item in top_features]

        fig, ax = plt.subplots(figsize=(12, 10))
        shap_values_top = shap_values[:, top_indices]
        X_analysis_top = X_analysis[:, top_indices]

        shap.summary_plot(
            shap_values_top,
            X_analysis_top,
            feature_names=top_names,
            show=False,
            max_display=15
        )
        plt.title(f"SHAP - Características Más Importantes\n{current_model_name} (Top 15 de {X_processed.shape[1]} características)", fontsize=14, pad=20)
        plt.tight_layout()

        print("✅ SHAP con nombres descriptivos completado")
        return fig

    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 6))
        error_text = f"❌ Error en SHAP con nombres:\n\n{str(e)[:200]}...\n\n🔄 Intentando método de backup..."
        ax.text(0.5, 0.5, error_text, ha='center', va='center', fontsize=10, 
               transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral"))
        ax.axis('off')
        print(f"❌ Error: {str(e)}")
 
        return fig


with gr.Blocks() as app:
    gr.Markdown("# 🏠 Predicción de Precios de Casas")
    
    with gr.Tabs():
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
            
            logs_table = gr.Dataframe()
            
            def fetch_logs_gradio():
                data = get_logs()
                if isinstance(data, dict) and "error" in data:
                    return pd.DataFrame([{"Error": data["error"]}])
                if not data:
                    return pd.DataFrame([{"Mensaje": "No hay registros disponibles"}])
                
                df = pd.DataFrame(data)
                return df
            
            fetch_logs_btn.click(fetch_logs_gradio, outputs=logs_table)


        with gr.TabItem("Análisis SHAP"):
            gr.Markdown("### Importancia de características (SHAP Summary Plot)")
            shap_btn = gr.Button("Generar Análisis SHAP")
            shap_plot = gr.Plot()
            shap_btn.click(get_shap_summary_plot, outputs=shap_plot)

if __name__ == "__main__":
    Thread(target=run_metrics_server, daemon=True).start()
    app.launch(server_name="0.0.0.0", server_port=8501)