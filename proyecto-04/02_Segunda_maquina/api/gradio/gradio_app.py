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

# 🏷️ VERSIÓN CON NOMBRES DE CARACTERÍSTICAS DESCRIPTIVOS

def get_shap_summary_plot():
    """SHAP TreeExplainer con nombres de características identificables"""
    
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
        print(f"🏷️ Iniciando SHAP con nombres de características para: {current_model_name}")
        
        # 🏷️ FUNCIÓN PARA OBTENER NOMBRES DE CARACTERÍSTICAS
        def get_feature_names_from_preprocessor():
            """Intenta obtener nombres reales del preprocesador"""
            try:
                preprocessor = load_preprocessor()
                
                if hasattr(preprocessor, 'get_feature_names_out'):
                    # Scikit-learn moderno
                    feature_names = preprocessor.get_feature_names_out()
                    return [str(name) for name in feature_names]
                elif hasattr(preprocessor, 'get_feature_names'):
                    # Scikit-learn antiguo
                    feature_names = preprocessor.get_feature_names()
                    return [str(name) for name in feature_names]
                else:
                    # Método manual para ColumnTransformer
                    return get_feature_names_manual(preprocessor)
            except Exception as e:
                print(f"⚠️ Error obteniendo nombres del preprocesador: {e}")
                return None
        
        def get_feature_names_manual(preprocessor):
            """Extrae nombres manualmente del ColumnTransformer"""
            try:
                feature_names = []
                
                if hasattr(preprocessor, 'transformers_'):
                    for name, transformer, features in preprocessor.transformers_:
                        if name == 'num':  # Transformador numérico
                            # Características numéricas mantienen sus nombres
                            feature_names.extend([f"num__{feat}" for feat in features])
                        elif name == 'cat':  # Transformador categórico  
                            # OneHotEncoder crea múltiples columnas por categoría
                            if hasattr(transformer, 'categories_'):
                                for i, feature in enumerate(features):
                                    categories = transformer.categories_[i]
                                    for category in categories:
                                        feature_names.append(f"cat__{feature}__{category}")
                
                return feature_names
            except Exception as e:
                print(f"⚠️ Error en extracción manual: {e}")
                return None
        
        # 🏷️ CREAR MAPEO DE CARACTERÍSTICAS IMPORTANTES
        def create_feature_mapping(feature_names, shap_values):
            """Crea mapeo de nombres más descriptivos para las características importantes"""
            
            # Características que sabemos que son importantes
            important_patterns = {
                # Numéricas
                'bed': 'Habitaciones',
                'bath': 'Baños', 
                'acre_lot': 'Terreno (acres)',
                'house_size': 'Tamaño casa (sqft)',
                'prev_sold_year': 'Año venta anterior',
                
                # Categóricas comunes
                'status__for_sale': 'Estado: En venta',
                'status__ready_to_build': 'Estado: Listo construir',
                'state__Connecticut': 'Estado: Connecticut',
                'state__New York': 'Estado: Nueva York',
                'city__East Windsor': 'Ciudad: East Windsor',
                'brokered_by': 'Agencia inmobiliaria'
            }
            
            # Calcular importancia de cada característica
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # Obtener índices de características más importantes
            top_indices = np.argsort(feature_importance)[-20:]  # Top 20
            
            mapped_names = []
            for idx in top_indices:
                original_name = feature_names[idx] if feature_names and idx < len(feature_names) else f"Feature_{idx}"
                
                # Buscar nombre más descriptivo
                descriptive_name = original_name
                for pattern, desc_name in important_patterns.items():
                    if pattern in original_name:
                        descriptive_name = desc_name
                        break
                
                # Si no encontramos mapeo, crear uno basado en el nombre original
                if descriptive_name == original_name and feature_names:
                    if 'num__' in original_name:
                        descriptive_name = original_name.replace('num__', '').replace('_', ' ').title()
                    elif 'cat__' in original_name:
                        parts = original_name.replace('cat__', '').split('__')
                        if len(parts) >= 2:
                            descriptive_name = f"{parts[0].title()}: {parts[1]}"
                        else:
                            descriptive_name = parts[0].title()
                
                mapped_names.append((idx, descriptive_name, feature_importance[idx]))
            
            # Ordenar por importancia (mayor a menor)
            mapped_names.sort(key=lambda x: x[2], reverse=True)
            
            return mapped_names
        
        # 🌳 Extraer modelo LightGBM y preparar datos (igual que antes)
        lightgbm_model = None
        try:
            if hasattr(model, '_model_impl'):
                if hasattr(model._model_impl, 'lgb_model'):
                    lightgbm_model = model._model_impl.lgb_model
                elif hasattr(model._model_impl, '_model'):
                    lightgbm_model = model._model_impl._model
            
            if lightgbm_model is None:
                try:
                    temp_explainer = shap.TreeExplainer(model)
                    lightgbm_model = model
                    print("✅ Modelo compatible directamente con TreeExplainer")
                except:
                    pass
        except Exception as e:
            print(f"⚠️ Error extrayendo modelo: {e}")
        
        if lightgbm_model is None:
            return get_shap_hybrid_method(model)
        
        # Preparar datos de muestra
        shap_samples = pd.DataFrame({
            'bed': [2, 3, 4, 5, 2],
            'bath': [1, 2, 3, 3, 1], 
            'acre_lot': [0.15, 0.25, 0.35, 0.45, 0.20],
            'house_size': [1000, 1500, 2000, 2500, 1200],
            'prev_sold_year': [2020, 2019, 2018, 2017, 2021]
        })
        
        # Crear datos completos y preprocesar
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
        
        print(f"🏷️ Datos preprocesados: {X_processed.shape}")
        
        # 🏷️ OBTENER NOMBRES DE CARACTERÍSTICAS
        print("🏷️ Obteniendo nombres de características...")
        feature_names = get_feature_names_from_preprocessor()
        
        if feature_names:
            print(f"✅ {len(feature_names)} nombres de características obtenidos")
        else:
            print("⚠️ No se pudieron obtener nombres, usando genéricos")
        
        # 🌳 CREAR TREE EXPLAINER Y CALCULAR SHAP
        print("🌳 Creando TreeExplainer...")
        explainer = shap.TreeExplainer(lightgbm_model)
        
        print("🌳 Calculando SHAP values...")
        X_analysis = X_processed[:3]  # Solo 3 muestras
        shap_values = explainer.shap_values(X_analysis)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        print(f"✅ SHAP values calculados: {shap_values.shape}")
        
        # 🏷️ CREAR MAPEO DE CARACTERÍSTICAS Y SELECCIONAR TOP
        mapped_features = create_feature_mapping(feature_names, shap_values)
        
        # Seleccionar top 15 características más importantes
        top_features = mapped_features[:15]
        top_indices = [item[0] for item in top_features]
        top_names = [item[1] for item in top_features]
        
        print(f"🏷️ Top características identificadas:")
        for i, (_, name, importance) in enumerate(top_features[:5]):
            print(f"  {i+1}. {name} (importancia: {importance:.4f})")
        
        # 🏷️ CREAR VISUALIZACIÓN CON NOMBRES DESCRIPTIVOS
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Usar solo las características top para el plot
        shap_values_top = shap_values[:, top_indices]
        X_analysis_top = X_analysis[:, top_indices]
        
        try:
            shap.summary_plot(
                shap_values_top,
                X_analysis_top,
                feature_names=top_names,
                show=False,
                max_display=15
            )
            
            plt.title(f"SHAP - Características Más Importantes\n{current_model_name} (Top 15 de {X_processed.shape[1]} características)", 
                     fontsize=14, pad=20)
            
        except Exception as plot_error:
            print(f"⚠️ Error en summary_plot: {plot_error}")
            # Plot alternativo
            feature_importance = np.abs(shap_values_top).mean(axis=0)
            
            plt.barh(range(len(top_names)), feature_importance)
            plt.yticks(range(len(top_names)), top_names)
            plt.xlabel("Importancia promedio |SHAP|")
            plt.title(f"Importancia de Características - {current_model_name}")
        
        plt.tight_layout()
        
        print("✅ SHAP con nombres descriptivos completado")
        return fig
        
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        error_text = f"❌ Error en SHAP con nombres:\n\n{str(e)[:200]}...\n\n"
        error_text += "🔄 Intentando método de backup..."
        
        ax.text(0.5, 0.5, error_text, ha='center', va='center', fontsize=10, 
               transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral"))
        ax.axis('off')
        
        print(f"❌ Error: {str(e)}")
        
        try:
            return get_shap_hybrid_method(loaded_models[current_model_name])
        except:
            return fig

# 🔄 MÉTODO HÍBRIDO MEJORADO (backup)
def get_shap_hybrid_method(model):
    """Método híbrido con nombres descriptivos"""
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("🔄 Usando método híbrido con nombres descriptivos...")
    
    try:
        base_data = {
            'bed': 3, 'bath': 2, 'acre_lot': 0.25, 'house_size': 1500, 'prev_sold_year': 2020,
            'brokered_by': '101640.0', 'status': 'for_sale', 'street': '1758218.0',
            'city': 'East Windsor', 'state': 'Connecticut', 'zip_code': '6016.0',
            'prev_sold_date': '2020-01-01'
        }
        
        # Predicción base
        base_df = pd.DataFrame([base_data])
        base_processed = preprocess_input(base_df)
        base_price = model.predict(base_processed)[0]
        
        # Variaciones con nombres descriptivos
        variations = {
            '🏠 Habitaciones (+1)': {'bed': 4},
            '🚿 Baños (+1)': {'bath': 3},
            '🌳 Terreno (+0.1 acres)': {'acre_lot': 0.35},
            '📐 Tamaño casa (+500 sqft)': {'house_size': 2000},
            '📅 Año más reciente (+3)': {'prev_sold_year': 2023},
            '🏢 Estado premium': {'state': 'New York'},
            '📍 Ciudad diferente': {'city': 'Hartford'}
        }
        
        feature_impacts = {}
        for feature_name, change in variations.items():
            try:
                mod_data = base_data.copy()
                mod_data.update(change)
                mod_df = pd.DataFrame([mod_data])
                mod_processed = preprocess_input(mod_df)
                impact = model.predict(mod_processed)[0] - base_price
                feature_impacts[feature_name] = impact
            except Exception as e:
                print(f"⚠️ Error en variación {feature_name}: {e}")
                continue
        
        # Crear visualización
        fig, ax = plt.subplots(figsize=(12, 8))
        
        features = list(feature_impacts.keys())
        impacts = list(feature_impacts.values())
        colors = ['green' if x > 0 else 'red' for x in impacts]
        
        bars = ax.barh(features, impacts, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('Impacto en precio (USD)')
        ax.set_title(f'Análisis de Importancia por Característica\n{current_model_name} (Baseline: ${base_price:,.0f})')
        
        # Agregar valores en las barras
        for bar, impact in zip(bars, impacts):
            width = bar.get_width()
            x_pos = width + (max(abs(min(impacts)), max(impacts)) * 0.02)
            if width < 0:
                x_pos = width - (max(abs(min(impacts)), max(impacts)) * 0.02)
                ha = 'right'
            else:
                ha = 'left'
            
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
                   f'${impact:,.0f}', ha=ha, va='center', fontweight='bold')
        
        plt.tight_layout()
        print("✅ Método híbrido con nombres completado")
        return fig
        
    except Exception as e:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"Error: {str(e)[:100]}...", ha='center', va='center')
        ax.axis('off')
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