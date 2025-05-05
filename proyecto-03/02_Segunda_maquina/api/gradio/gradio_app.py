import os
import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
import gradio as gr
import traceback
import numpy as np
import requests
import json

# Configuración de MLFlow
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://10.43.101.206:30382"
os.environ['AWS_ACCESS_KEY_ID'] = "adminuser"
os.environ['AWS_SECRET_ACCESS_KEY'] = "securepassword123"
mlflow.set_tracking_uri("http://10.43.101.206:30500")

# Variable para almacenar los modelos cargados
loaded_models = {}
current_model_name = None

# Función para obtener modelos de MLflow
def list_mlflow_models():
    try:
        client = mlflow.tracking.MlflowClient()
        registered_models = client.search_registered_models()
        
        production_models = []
        
        for model in registered_models:
            model_info = {
                "name": model.name,
                "versions": [],
                "lastUpdated": model.last_updated_timestamp
            }
            
            # Filtrar solo versiones en producción
            for version in client.search_model_versions(f"name='{model.name}'"):
                if version.current_stage == "Production":
                    model_info["versions"].append({
                        "version": version.version,
                        "creation_timestamp": version.creation_timestamp
                    })
            
            if model_info["versions"]:  # Añadir modelo si tiene versiones en producción
                production_models.append(model_info)
        
        return production_models
    
    except Exception as e:
        print(f"Error al listar modelos en producción: {e}")
        return {"success": False, "error": str(e)}


def load_model(model_name):
    global current_model_name, loaded_models
    
    if not model_name:
        return " Por favor, seleccione un modelo para cargar."
    
    try:
        if model_name in loaded_models:
            current_model_name = model_name
            print(f"Modelo '{model_name}' ya cargado y seleccionado.")
            return f" Modelo '{model_name}' seleccionado."
        
        print(f"=== INICIO DE CARGA DE MODELO: {model_name} ===")
        
        try:
            client = mlflow.tracking.MlflowClient()
            
            versions = client.search_model_versions(f"name='{model_name}'")
            production_versions = [v for v in versions if v.current_stage == "Production"]
            
            if not production_versions:
                return f" No se encontró una versión en producción para el modelo '{model_name}'."
            
            latest_prod_version = sorted(production_versions, key=lambda x: int(x.version), reverse=True)[0]
            
            print(f"Cargando versión de producción: {latest_prod_version.version} para {model_name}")
            print(f"Run ID: {latest_prod_version.run_id}")
            print(f"Source: {latest_prod_version.source}")
            
            model_uri = f"models:/{model_name}/Production"
            print(f"Intentando cargar modelo con URI: {model_uri}")
            
            # Ahora cargamos el modelo directamente, ya no espera un pipeline
            loaded_models[model_name] = mlflow.pyfunc.load_model(model_uri)
            current_model_name = model_name
            
            print(f"Modelo '{model_name}' cargado exitosamente")
            
            try:
                # También necesitamos cargar información de columnas esperadas
                run_id = latest_prod_version.run_id
                
                # Intentar obtener artefacto con información de columnas
                artifact_uri = client.get_run(run_id).info.artifact_uri
                columns_path = f"{artifact_uri}/{model_name}_columns.json"
                
                print(f"Verificando si existe artefacto con lista de columnas: {columns_path}")
                
                # También intentamos una prueba simple
                test_df = pd.DataFrame({
                    "race": ["Caucasian"],
                    "gender": ["Male"],
                    "age": ["[50-60)"],
                    "admission_type_id": [1],
                    "discharge_disposition_id": [1],
                    "admission_source_id": [1],
                    "time_in_hospital": [4],
                    "num_lab_procedures": [45],
                    "num_procedures": [1],
                    "num_medications": [16],
                    "number_outpatient": [0],
                    "number_emergency": [0],
                    "number_inpatient": [0],
                    "diag_1": ["250.00"],
                    "diag_2": ["250.00"],
                    "diag_3": ["250.00"],
                    "number_diagnoses": [7],
                    "max_glu_serum": ["None"],
                    "A1Cresult": ["None"],
                    "insulin": ["No"],
                    "diabetesMed": ["Yes"],
                    "metformin": ["No"],
                    "repaglinide": ["No"],
                    "nateglinide": ["No"],
                    "chlorpropamide": ["No"],
                    "glimepiride": ["No"],
                    "acetohexamide": ["No"],
                    "glipizide": ["No"],
                    "glyburide": ["No"],
                    "tolbutamide": ["No"],
                    "pioglitazone": ["No"],
                    "rosiglitazone": ["No"],
                    "acarbose": ["No"],
                    "miglitol": ["No"],
                    "troglitazone": ["No"],
                    "tolazamide": ["No"],
                    "examide": ["No"],
                    "citoglipton": ["No"],
                    "glyburide-metformin": ["No"],
                    "glipizide-metformin": ["No"],
                    "glimepiride-pioglitazone": ["No"],
                    "metformin-rosiglitazone": ["No"],
                    "metformin-pioglitazone": ["No"]
                })
                
                # Codificar categorías para prueba
                cat_columns = test_df.select_dtypes(include=['object']).columns
                test_encoded = pd.get_dummies(test_df, columns=cat_columns, drop_first=False)
                
                print("Realizando predicción de prueba...")
                test_pred = loaded_models[model_name].predict(test_encoded)
                print(f"Predicción de prueba exitosa: {test_pred}")
            except Exception as test_err:
                print(f"Error en prueba o carga de metadatos: {str(test_err)}")
                print(traceback.format_exc())
            
            return f" Modelo '{model_name}' cargado y seleccionado correctamente."
            
        except Exception as e:
            print(f"Error al cargar modelo: {type(e).__name__}: {str(e)}")
            print(f"Traza completa: {traceback.format_exc()}")
            
            try:
                print("Intentando método de carga alternativo por nombre/etapa...")
                model_uri = f"models:/{model_name}/Production"
                loaded_models[model_name] = mlflow.pyfunc.load_model(model_uri)
                current_model_name = model_name
                return f" Modelo '{model_name}' cargado y seleccionado correctamente con método alternativo."
            except Exception as alt_error:
                print(f"Error en método alternativo: {str(alt_error)}")
            
            if "404" in str(e) or "Not Found" in str(e):
                return f" No se pudo cargar el modelo '{model_name}'. No se encontró el archivo del modelo en el almacenamiento S3. Verifique la ruta y las credenciales de S3."
            else:
                return f" No se pudo cargar el modelo '{model_name}'. Error: {str(e)}"
    except Exception as e:
        print(f"=== ERROR DE CARGA DE MODELO ===")
        print(f"Error detallado: {type(e).__name__}: {str(e)}")
        print(traceback.format_exc())
        
        return f"""
        <div style="background-color: #ffebee; padding: 15px; border-radius: 10px; border-left: 5px solid #f44336;">
            <h3 style="color: #c62828; margin-top: 0;"> Error al cargar el modelo</h3>
            <p><strong>Descripción:</strong> {str(e)}</p>
            <p><strong>Tipo de error:</strong> {type(e).__name__}</p>
            <details>
                <summary>Detalles técnicos</summary>
                <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 12px;">
{traceback.format_exc()}
                </pre>
            </details>
        </div>
        """
           
# Función para realizar predicciones
def predict(
    model_name,
    race,
    gender,
    age,
    admission_type_id,
    time_in_hospital,
    num_lab_procedures,
    num_procedures,
    num_medications,
    number_outpatient,
    number_emergency,
    number_inpatient,
    number_diagnoses,
    max_glu_serum,
    A1Cresult,
    insulin,
    diabetesMed
):
    global current_model_name, loaded_models
    
    if not current_model_name and not model_name:
        return " No hay un modelo seleccionado. Por favor, seleccione un modelo antes de realizar predicciones."
    
    model_to_use = model_name if model_name else current_model_name
    
    if model_to_use not in loaded_models:
        return f" El modelo '{model_to_use}' no está cargado. Por favor, selecciónelo primero."
    
    try:
        input_data = pd.DataFrame({
            'race': [race],
            'gender': [gender],
            'age': [age],
            'admission_type_id': [int(admission_type_id)],
            'discharge_disposition_id': [1],
            'admission_source_id': [1],
            'time_in_hospital': [int(time_in_hospital)],
            'num_lab_procedures': [int(num_lab_procedures)],
            'num_procedures': [int(num_procedures)],
            'num_medications': [int(num_medications)],
            'number_outpatient': [int(number_outpatient)],
            'number_emergency': [int(number_emergency)],
            'number_inpatient': [int(number_inpatient)],
            'number_diagnoses': [int(number_diagnoses)],
            'diag_1': ['250.00'],
            'diag_2': ['250.00'],
            'diag_3': ['250.00'],
            'max_glu_serum': [max_glu_serum],
            'A1Cresult': [A1Cresult],
            'insulin': [insulin],
            'diabetesMed': [diabetesMed]
        })
        
        medication_columns = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
            'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
            'miglitol', 'troglitazone', 'tolazamide', 'examide',
            'citoglipton', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]
        
        for med in medication_columns:
            input_data[med] = 'No'
        
        cat_columns = input_data.select_dtypes(include=['object']).columns
        
        input_data_encoded = pd.get_dummies(input_data, columns=cat_columns, drop_first=False)
        
        # Intento obtener info del modelo
        try:
            # Intentar obtener las columnas que el modelo espera
            model_api_url = f"{mlflow.get_tracking_uri()}/api/2.0/mlflow/registered-models/get?name={model_to_use}"
            response = requests.get(model_api_url)
            if response.status_code == 200:
                model_info = response.json()
                if 'registered_model' in model_info:
                    model_versions = model_info['registered_model'].get('latest_versions', [])
                    if model_versions:
                        # Buscar versión en Producción
                        prod_versions = [v for v in model_versions if v.get('current_stage') == 'Production']
                        if prod_versions:
                            latest_version = prod_versions[0]
                            run_id = latest_version.get('run_id')
                            if run_id:
                                # Intentar obtener columnas desde los artefactos
                                artifacts_api_url = f"{mlflow.get_tracking_uri()}/api/2.0/mlflow/artifacts/list?run_id={run_id}&path={model_to_use}_columns.json"
                                artifacts_response = requests.get(artifacts_api_url)
                                if artifacts_response.status_code == 200 and 'files' in artifacts_response.json():
                                    # Las columnas esperadas están guardadas en un archivo JSON
                                    try:
                                        # Descargar el archivo JSON
                                        column_file_url = f"{mlflow.get_tracking_uri()}/get-artifact?run_id={run_id}&path={model_to_use}_columns.json"
                                        column_response = requests.get(column_file_url)
                                        if column_response.status_code == 200:
                                            expected_columns = json.loads(column_response.text)
                                            
                                            # Asegurar que todas las columnas esperadas existan
                                            for col in expected_columns:
                                                if col not in input_data_encoded.columns:
                                                    input_data_encoded[col] = 0
                                            
                                            # Ordenar columnas como el modelo espera
                                            input_data_encoded = input_data_encoded[expected_columns]
                                    except Exception as e:
                                        print(f"Error obteniendo columnas esperadas: {e}")
        except Exception as e:
            print(f"Error consultando MLflow API: {e}")
        
        # Realizar predicción
        model = loaded_models[model_to_use]
        prediction = model.predict(input_data_encoded)
        
        readmitted_types = {
            "NO": "No readmitido",
            "<30": "Readmitido en menos de 30 días",
            ">30": "Readmitido después de 30 días"
        }
        
        try:
            pred_value = str(prediction[0])
            readmitted_type = readmitted_types.get(pred_value, f"Tipo {pred_value}")
        except Exception as inner_e:
            print(f"Error al interpretar el resultado: {str(inner_e)}")
            readmitted_type = str(prediction[0])
        
        result = f"""
        <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50;">
            <h3 style="color: #2e7d32; margin-top: 0;"> Resultado de la Predicción</h3>
            <div style="font-size: 18px; margin-bottom: 15px;">
                <strong>Readmisión del paciente:</strong> <span style="background-color: #81c784; padding: 5px 10px; border-radius: 5px; color: white;">{readmitted_type} ({prediction[0]})</span>
            </div>
            <div style="font-size: 16px; margin-bottom: 15px;">
                <strong>Modelo utilizado:</strong> {model_to_use}
            </div>
            
            <div style="margin-top: 20px;">
                <h4 style="color: #2e7d32;"> Datos de entrada</h4>
                <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                    <tr style="background-color: #c8e6c9;">
                        <th style="padding: 8px; text-align: left; border: 1px solid #a5d6a7;">Característica</th>
                        <th style="padding: 8px; text-align: left; border: 1px solid #a5d6a7;">Valor</th>
                    </tr>
        """
        
        input_data_display = {
            "Raza": race,
            "Género": gender,
            "Grupo de edad": age,
            "Tipo de admisión": admission_type_id,
            "Tiempo en hospital (días)": time_in_hospital,
            "Procedimientos de laboratorio": num_lab_procedures,
            "Procedimientos": num_procedures,
            "Número de medicamentos": num_medications,
            "Visitas ambulatorias": number_outpatient,
            "Visitas a emergencia": number_emergency,
            "Hospitalizaciones previas": number_inpatient,
            "Número de diagnósticos": number_diagnoses,
            "Nivel máximo de glucosa sérica": max_glu_serum,
            "Resultado de HbA1c": A1Cresult,
            "Cambio en dosis de insulina": insulin,
            "Medicamento para diabetes": diabetesMed
        }
        
        for key, value in input_data_display.items():
            result += f"""
                    <tr>
                        <td style="padding: 8px; border: 1px solid #a5d6a7;">{key}</td>
                        <td style="padding: 8px; border: 1px solid #a5d6a7;">{value}</td>
                    </tr>
            """
        
        result += """
                </table>
            </div>
        </div>
        """
        
        return result
    
    except Exception as e:
        print(f"=== ERROR EN PREDICCIÓN ===")
        print(f"Error detallado: {type(e).__name__}: {str(e)}")
        print(traceback.format_exc())
        
        return f"""
        <div style="background-color: #ffebee; padding: 15px; border-radius: 10px; border-left: 5px solid #f44336;">
            <h3 style="color: #c62828; margin-top: 0;"> Error en la predicción</h3>
            <p><strong>Descripción:</strong> {str(e)}</p>
            <p><strong>Tipo de error:</strong> {type(e).__name__}</p>
            <details>
                <summary>Detalles técnicos</summary>
                <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 12px;">
{traceback.format_exc()}
                </pre>
            </details>
            <p><strong>Recomendación:</strong> Verifique que el formato de los datos de entrada coincida con el formato esperado por el modelo. Si el problema persiste, considere reentrenar el modelo con un preprocesamiento más robusto.</p>
        </div>
        """
 
# Función para obtener información de modelos
def refresh_models():
    models = list_mlflow_models()
    model_names = []
    model_info = ""

    if models:
        model_info = """
        <div style="background-color: #e8f0fe; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="color: #1976d2; margin-top: 0;">Modelos Disponibles en Producción</h3>
        """
        
        for model in models:
            # Añadir nombre de modelo a la lista
            model_names.append(model["name"])
            
            production_badge = """ <span style="background-color: #4caf50; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px;">PRODUCCIÓN</span>"""
            versions_count = len(model["versions"]) if "versions" in model else 0
            
            model_info += f"""
            <div style="background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #1976d2;">
                <h4 style="margin-top: 0; margin-bottom: 10px; color: #1976d2;">{model["name"]}{production_badge}</h4>
                <p style="margin: 5px 0;"><strong>Versiones:</strong> {versions_count}</p>
            </div>
            """
        
        model_info += """
        </div>
        """
    else:
        model_info = """
        <div style="background-color: #fff3e0; padding: 15px; border-radius: 10px; border-left: 5px solid #ff9800;">
            <h3 style="color: #e65100; margin-top: 0;"> Sin Modelos Disponibles en Producción</h3>
            <p>No se encontraron modelos registrados en el estado de Producción en MLflow. Verifique la conexión con el servidor MLflow.</p>
        </div>
        """
    
    # Utilizar update() para actualizar correctamente el dropdown
    return gr.Dropdown(choices=model_names, value=None if not model_names else model_names[0]), model_info

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont("Roboto"), "ui-sans-serif", "system-ui", "sans-serif"],
    spacing_size=gr.themes.sizes.spacing_md,
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#f9f9f9",
    body_background_fill_dark="#1a1a1a",
    button_primary_background_fill="#1976d2",
    button_primary_background_fill_hover="#1565c0",
    button_primary_text_color="white",
    button_secondary_background_fill="#e3f2fd",
    button_secondary_background_fill_hover="#bbdefb",
    button_secondary_text_color="#1976d2",
    block_title_text_color="#1976d2",
    block_label_text_color="#555",
    input_background_fill="#fff",
    input_border_color="#ddd",
    input_shadow="0 2px 4px rgba(0,0,0,0.05)",
    checkbox_background_color="#2196f3",
    slider_color="#2196f3",
    slider_color_dark="#64b5f6",
)

# Imágenes para el diseño visual
header_html = """
<div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
    <img src="https://cdn-icons-png.flaticon.com/512/2966/2966327.png" height="60px" style="margin-right: 20px;">
    <div>
        <h1 style="margin: 0; color: #1976d2; font-size: 28px;">Predictor de Readmisión de Pacientes Diabéticos</h1>
        <p style="margin: 5px 0 0; color: #555; font-size: 16px;">Modelo de aprendizaje automático para hospitales y centros de salud</p>
    </div>
</div>
"""

footer_html = """
<div style="margin-top: 30px; text-align: center; border-top: 1px solid #ddd; padding-top: 20px;">
    <p style="color: #555; font-size: 14px;">Sistema de Predicción de Readmisión Hospitalaria con MLflow y Gradio © 2025</p>
    <p style="color: #777; font-size: 12px;">Desarrollado para la mejora de la atención médica y gestión de pacientes diabéticos</p>
    <div style="display: flex; justify-content: center; gap: 15px; margin-top: 10px;">
        <div style="display: flex; align-items: center;">
            <span style="background-color: #2196f3; color: white; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></span>
            <span style="margin-left: 5px; color: #555; font-size: 12px;">MLflow</span>
        </div>
        <div style="display: flex; align-items: center;">
            <span style="background-color: #673ab7; color: white; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></span>
            <span style="margin-left: 5px; color: #555; font-size: 12px;">Gradio</span>
        </div>
        <div style="display: flex; align-items: center;">
            <span style="background-color: #ff9800; color: white; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></span>
            <span style="margin-left: 5px; color: #555; font-size: 12px;">Python</span>
        </div>
    </div>
</div>
"""

# Configuración de la aplicación Gradio con el tema personalizado
with gr.Blocks(theme=theme) as app:
    # Encabezado
    gr.HTML(header_html)
    
    # Panel de selección de modelo
    with gr.Tab("1️⃣ Selección de Modelo"):
        gr.Markdown("### Seleccione un modelo en Producción")
        
        with gr.Row():
            with gr.Column(scale=3):
                # Refresh button to map models
                refresh_button = gr.Button(" Mapear Modelos", variant="primary", size="lg")
                model_info = gr.HTML("*Haga clic en 'Mapear Modelos' para ver los modelos disponibles.*")
            
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("### 💻 Carga de Modelo")
                    model_dropdown = gr.Dropdown(
                        label="Seleccione un modelo",
                        choices=[],  # Initially empty, will be updated after refresh
                        interactive=True
                    )
                    load_button = gr.Button(" Cargar Modelo", variant="primary")
                    load_output = gr.HTML()

        # Link the refresh button to fetch models and populate the dropdown
        refresh_button.click(refresh_models, outputs=[model_dropdown, model_info])

        # Link the load button to load the selected model
        load_button.click(load_model, inputs=model_dropdown, outputs=load_output)


    # Panel de predicción
    with gr.Tab("2️⃣ Realizar Predicción"):
        gr.Markdown("### Ingrese los datos del paciente para la predicción")
        
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown("#### 👤 Información demográfica")
                    race = gr.Dropdown(
                        label="Raza",
                        choices=["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"],
                        value="Caucasian",
                        info="Grupo étnico del paciente"
                    )
                    
                    gender = gr.Dropdown(
                        label="Género",
                        choices=["Male", "Female", "Unknown"],
                        value="Male",
                        info="Género del paciente"
                    )
                    
                    age = gr.Dropdown(
                        label="Grupo de edad",
                        choices=["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", 
                                 "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
                        value="[50-60)",
                        info="Rango de edad del paciente"
                    )
                    
                    admission_type_id = gr.Slider(
                        label="Tipo de admisión",
                        minimum=1,
                        maximum=8,
                        value=1,
                        step=1,
                        info="Tipo de admisión hospitalaria (1=Emergencia, 2=Urgente, 3=Electiva, etc.)"
                    )
            
            with gr.Column():
                with gr.Group():
                    gr.Markdown("#### 🏥 Historial médico")
                    time_in_hospital = gr.Slider(
                        label="Tiempo en hospital (días)",
                        minimum=1,
                        maximum=14,
                        value=4,
                        step=1,
                        info="Duración de la estadía hospitalaria"
                    )
                    
                    num_lab_procedures = gr.Slider(
                        label="Número de procedimientos de laboratorio",
                        minimum=1,
                        maximum=120,
                        value=45,
                        step=1,
                        info="Cantidad de pruebas de laboratorio realizadas"
                    )
                    
                    num_procedures = gr.Slider(
                        label="Número de procedimientos",
                        minimum=0,
                        maximum=6,
                        value=1,
                        step=1,
                        info="Cantidad de procedimientos (no de laboratorio) realizados"
                    )
                    
                    num_medications = gr.Slider(
                        label="Número de medicamentos",
                        minimum=1,
                        maximum=81,
                        value=16,
                        step=1,
                        info="Cantidad total de medicamentos administrados"
                    )
            
            with gr.Column():
                with gr.Group():
                    gr.Markdown("#### 🔄 Historial de visitas y pruebas")
                    number_outpatient = gr.Slider(
                        label="Visitas ambulatorias",
                        minimum=0,
                        maximum=42,
                        value=0,
                        step=1,
                        info="Número de visitas ambulatorias en el año previo"
                    )
                    
                    number_emergency = gr.Slider(
                                            label="Visitas a emergencia",
                                            minimum=0,
                                            maximum=76,
                                            value=0,
                                            step=1,
                                            info="Número de visitas a emergencia en el año previo"
                                        )
                                        
                    number_inpatient = gr.Slider(
                        label="Hospitalizaciones previas",
                        minimum=0,
                        maximum=21,
                        value=0,
                        step=1,
                        info="Número de hospitalizaciones en el año previo"
                    )
                    
                    number_diagnoses = gr.Slider(
                        label="Número de diagnósticos",
                        minimum=1,
                        maximum=16,
                        value=7,
                        step=1,
                        info="Número de diagnósticos registrados durante esta hospitalización"
                    )
        
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown("#### 🧪 Pruebas y tratamiento de diabetes")
                    max_glu_serum = gr.Dropdown(
                        label="Nivel máximo de glucosa sérica",
                        choices=["None", "Norm", ">200", ">300"],
                        value="None",
                        info="Resultado de la prueba de glucosa sérica"
                    )
                    
                    A1Cresult = gr.Dropdown(
                        label="Resultado de HbA1c",
                        choices=["None", "Norm", ">7", ">8"],
                        value="None",
                        info="Resultado de la prueba de hemoglobina A1c"
                    )
                    
                    insulin = gr.Dropdown(
                        label="Cambio en dosis de insulina",
                        choices=["No", "Up", "Down", "Steady"],
                        value="No",
                        info="Cambio en la dosis de insulina durante la hospitalización"
                    )
                    
                    diabetesMed = gr.Dropdown(
                        label="¿Se prescribió medicamento para diabetes?",
                        choices=["Yes", "No"],
                        value="Yes",
                        info="Si se prescribió algún medicamento para diabetes"
                    )
        
        # Panel de resultados
        with gr.Row():
            predict_button = gr.Button(" Realizar Predicción", variant="primary", size="lg")
        
        with gr.Row():
            prediction_output = gr.HTML()
    
    # Panel de información
    with gr.Tab("ℹ️ Información"):
        gr.Markdown("""
        # Acerca del Predictor de Readmisión de Pacientes Diabéticos
        
        Esta aplicación utiliza modelos de aprendizaje automático para predecir la probabilidad de readmisión hospitalaria de pacientes diabéticos dentro de los 30 días posteriores al alta.
        
        ## Tipos de Resultado
        
        Los modelos pueden predecir tres tipos de resultados:
        
        1. **NO** - El paciente no será readmitido
        2. **<30** - El paciente será readmitido en menos de 30 días
        3. **>30** - El paciente será readmitido después de 30 días
        
        ## Variables predictoras
        
        Las variables utilizadas para la predicción corresponden a datos demográficos y clínicos:
        
        - **Datos demográficos** - Raza, género y grupo de edad
        - **Información de admisión** - Tipo de admisión, tiempo de estadía
        - **Procedimientos médicos** - Número de procedimientos de laboratorio y no laboratorio
        - **Medicamentos** - Número total de medicamentos administrados
        - **Historial de visitas** - Visitas ambulatorias, de emergencia y hospitalizaciones previas
        - **Información sobre diabetes** - Resultados de pruebas de glucosa, HbA1c y tratamiento con insulina
        
        ## Conjunto de datos
        
        Este proyecto está basado en el conjunto de datos "Diabetes 130-US hospitals for years 1999-2008" que representa 10 años de atención clínica en 130 hospitales de EE.UU. El conjunto contiene más de 50 características que representan los resultados del paciente y del hospital.
        
        ## Desarrollo y tecnologías
        
        Esta aplicación está desarrollada con:
        - **MLflow** - Para la gestión y despliegue de modelos
        - **Gradio** - Para la interfaz de usuario
        - **Python** - Como lenguaje de programación base
        - **Kubernetes** - Para la orquestación de contenedores
        - **AirFlow** - Para la orquestación de flujos de trabajo
        - **PostgreSQL** - Para el almacenamiento de datos
        - **FastAPI** - Para el desarrollo de APIs
        """)
    
    # Pie de página
    gr.HTML(footer_html)
    
    predict_button.click(
        fn=predict,
        inputs=[
            model_dropdown,
            race,
            gender,
            age,
            admission_type_id,
            time_in_hospital,
            num_lab_procedures,
            num_procedures,
            num_medications,
            number_outpatient,
            number_emergency,
            number_inpatient,
            number_diagnoses,
            max_glu_serum,
            A1Cresult,
            insulin,
            diabetesMed
        ],
        outputs=[prediction_output]
    )

app.launch(
    share=False,
    server_name="0.0.0.0",
    server_port=8501,
    favicon_path="https://cdn-icons-png.flaticon.com/512/2966/2966327.png"
)
