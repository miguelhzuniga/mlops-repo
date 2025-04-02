import os
import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
import gradio as gr

# Configuración de MLFlow
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://10.43.101.202:9000"
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'
mlflow.set_tracking_uri("http://10.43.101.202:5000")

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
            
            # Filter only production versions
            for version in client.search_model_versions(f"name='{model.name}'"):
                if version.current_stage == "Production":
                    model_info["versions"].append({
                        "version": version.version,
                        "creation_timestamp": version.creation_timestamp
                    })
            
            if model_info["versions"]:  # Add model if it has production versions
                production_models.append(model_info)
        
        return production_models
    
    except Exception as e:
        print(f"Error al listar modelos en producción: {e}")
        return {"success": False, "error": str(e)}

# Función para cargar modelo
def load_model(model_name):
    global current_model_name, loaded_models
    
    if not model_name:
        return " Por favor, seleccione un modelo para cargar."
    
    try:
        # Verificar si el modelo ya está cargado
        if model_name in loaded_models:
            current_model_name = model_name
            return f" Modelo '{model_name}' seleccionado."
        
        # Cargar el modelo desde MLflow
        try:
            model_uri = f"models:/{model_name}/production"
            loaded_models[model_name] = mlflow.pyfunc.load_model(model_uri=model_uri)
            current_model_name = model_name
            return f" Modelo '{model_name}' cargado y seleccionado correctamente."
        except MlflowException as e:
            return f" No se pudo cargar el modelo '{model_name}'. Verifique que exista y tenga una versión en producción."
    except Exception as e:
        return f" Error al seleccionar modelo: {str(e)}"

# Función para realizar predicciones
def predict(
    model_name,
    elevation,
    aspect,
    slope,
    horizontal_distance_hydrology,
    vertical_distance_hydrology,
    horizontal_distance_roadways,
    hillshade_9am,
    hillshade_noon,
    hillshade_3pm,
    horizontal_distance_fire,
    wilderness_area,
    soil_type
):
    global current_model_name, loaded_models
    
    # Verificar si se ha seleccionado un modelo
    if not current_model_name and not model_name:
        return " No hay un modelo seleccionado. Por favor, seleccione un modelo antes de realizar predicciones."
    
    # Usar el modelo especificado o el actual
    model_to_use = model_name if model_name else current_model_name
    
    if model_to_use not in loaded_models:
        return f" El modelo '{model_to_use}' no está cargado. Por favor, selecciónelo primero."
    
    try:
        # Crear el diccionario de datos de entrada
        input_data = {
            "Elevation": int(elevation),
            "Aspect": int(aspect),
            "Slope": int(slope),
            "Horizontal_Distance_To_Hydrology": int(horizontal_distance_hydrology),
            "Vertical_Distance_To_Hydrology": int(vertical_distance_hydrology),
            "Horizontal_Distance_To_Roadways": int(horizontal_distance_roadways),
            "Hillshade_9am": int(hillshade_9am),
            "Hillshade_Noon": int(hillshade_noon),
            "Hillshade_3pm": int(hillshade_3pm),
            "Horizontal_Distance_To_Fire_Points": int(horizontal_distance_fire),
            "Wilderness_Area": wilderness_area,
            "Soil_Type": soil_type
        }
        
        # Convertir a dataframe para la predicción
        input_df = pd.DataFrame([input_data])
        
        # Realizar la predicción
        prediction = loaded_models[model_to_use].predict(input_df)
        
        # Mapeo de clases para interpretación visual
        cover_types = {
            1: "Spruce/Fir",
            2: "Lodgepole Pine",
            3: "Ponderosa Pine",
            4: "Cottonwood/Willow",
            5: "Aspen",
            6: "Douglas-fir",
            7: "Krummholz"
        }
        
        # Determinar el tipo de cobertura forestal
        try:
            pred_value = int(prediction[0])
            cover_type = cover_types.get(pred_value, f"Tipo {pred_value}")
        except:
            cover_type = str(prediction[0])
        
        # Construir la respuesta
        result = f"""
        <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50;">
            <h3 style="color: #2e7d32; margin-top: 0;"> Resultado de la Predicción</h3>
            <div style="font-size: 18px; margin-bottom: 15px;">
                <strong>Clase de cobertura forestal:</strong> <span style="background-color: #81c784; padding: 5px 10px; border-radius: 5px; color: white;">{cover_type} (Clase {prediction[0]})</span>
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
        
        # Agregar cada característica ingresada
        for key, value in input_data.items():
            formatted_key = key.replace("_", " ")
            result += f"""
                    <tr>
                        <td style="padding: 8px; border: 1px solid #a5d6a7;">{formatted_key}</td>
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
        return f"""
        <div style="background-color: #ffebee; padding: 15px; border-radius: 10px; border-left: 5px solid #f44336;">
            <h3 style="color: #c62828; margin-top: 0;"> Error en la predicción</h3>
            <p>{str(e)}</p>
        </div>
        """

# Función corregida para obtener información de modelos
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
    return gr.Dropdown.update(choices=model_names, value=None if not model_names else model_names[0]), model_info


# Configurar el tema personalizado para la aplicación
theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="blue",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont("Roboto"), "ui-sans-serif", "system-ui", "sans-serif"],
    spacing_size=gr.themes.sizes.spacing_md,
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#f9f9f9",
    body_background_fill_dark="#1a1a1a",
    button_primary_background_fill="#2e7d32",
    button_primary_background_fill_hover="#388e3c",
    button_primary_text_color="white",
    button_secondary_background_fill="#e8f5e9",
    button_secondary_background_fill_hover="#c8e6c9",
    button_secondary_text_color="#2e7d32",
    block_title_text_color="#2e7d32",
    block_label_text_color="#555",
    input_background_fill="#fff",
    input_border_color="#ddd",
    input_shadow="0 2px 4px rgba(0,0,0,0.05)",
    checkbox_background_color="#4caf50",
    slider_color="#4caf50",
    slider_color_dark="#81c784",
)

# Imágenes para el diseño visual
header_html = """
<div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
    <img src="https://cdn-icons-png.flaticon.com/512/2361/2361192.png" height="60px" style="margin-right: 20px;">
    <div>
        <h1 style="margin: 0; color: #2e7d32; font-size: 28px;">Predictor de Cobertura Forestal</h1>
        <p style="margin: 5px 0 0; color: #555; font-size: 16px;">Modelo de aprendizaje automático para análisis de ecosistemas forestales</p>
    </div>
</div>
"""

footer_html = """
<div style="margin-top: 30px; text-align: center; border-top: 1px solid #ddd; padding-top: 20px;">
    <p style="color: #555; font-size: 14px;">Sistema de Predicción de Cobertura Forestal con MLflow y Gradio © 2025</p>
    <p style="color: #777; font-size: 12px;">Desarrollado para la conservación y gestión sostenible de bosques</p>
    <div style="display: flex; justify-content: center; gap: 15px; margin-top: 10px;">
        <div style="display: flex; align-items: center;">
            <span style="background-color: #4caf50; color: white; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></span>
            <span style="margin-left: 5px; color: #555; font-size: 12px;">MLflow</span>
        </div>
        <div style="display: flex; align-items: center;">
            <span style="background-color: #2196f3; color: white; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></span>
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
                with gr.Group():  # Cambiado de Box a Group
                    gr.Markdown("### 烙 Carga de Modelo")
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
        gr.Markdown("### Ingrese los datos para la predicción")
        
        with gr.Row():
            with gr.Column():
                with gr.Group():  # Cambiado de Box a Group
                    gr.Markdown("#### ️ Características geográficas")
                    elevation = gr.Slider(
                        label="Elevación (m)",
                        minimum=0,
                        maximum=4000,
                        value=2500,
                        step=10,
                        info="Altura sobre el nivel del mar"
                    )
                    
                    aspect = gr.Slider(
                        label="Aspecto",
                        minimum=0,
                        maximum=360,
                        value=180,
                        step=1,
                        info="Orientación de la pendiente (0-360°), 0=Norte, 90=Este"
                    )
                    
                    slope = gr.Slider(
                        label="Pendiente",
                        minimum=0,
                        maximum=90,
                        value=15,
                        step=1,
                        info="Grado de inclinación del terreno (0-90°)"
                    )
                    
                    horizontal_distance_hydrology = gr.Slider(
                        label="Distancia horizontal a hidrología (m)",
                        minimum=0,
                        maximum=1500,
                        value=250,
                        step=10,
                        info="Distancia horizontal a ríos, lagos o cuerpos de agua"
                    )
            
            with gr.Column():
                with gr.Group():  # Cambiado de Box a Group
                    gr.Markdown("####  Distancias y terreno")
                    vertical_distance_hydrology = gr.Slider(
                        label="Distancia vertical a hidrología (m)",
                        minimum=-500,
                        maximum=500,
                        value=45,
                        step=5,
                        info="Diferencia vertical con respecto al cuerpo de agua más cercano"
                    )
                    
                    horizontal_distance_roadways = gr.Slider(
                        label="Distancia a carreteras (m)",
                        minimum=0,
                        maximum=8000,
                        value=1200,
                        step=50,
                        info="Distancia horizontal a la carretera más cercana"
                    )
                    
                    hillshade_9am = gr.Slider(
                        label="Sombra de colina 9am",
                        minimum=0,
                        maximum=255,
                        value=220,
                        step=1,
                        info="Índice de iluminación a las 9:00 (0-255)"
                    )
                    
                    hillshade_noon = gr.Slider(
                        label="Sombra de colina mediodía",
                        minimum=0,
                        maximum=255,
                        value=235,
                        step=1,
                        info="Índice de iluminación al mediodía (0-255)"
                    )
            
            with gr.Column():
                with gr.Group():  # Cambiado de Box a Group
                    gr.Markdown("####  Condiciones y clasificación")
                    hillshade_3pm = gr.Slider(
                        label="Sombra de colina 3pm",
                        minimum=0,
                        maximum=255,
                        value=180,
                        step=1,
                        info="Índice de iluminación a las 15:00 (0-255)"
                    )
                    
                    horizontal_distance_fire = gr.Slider(
                        label="Distancia a puntos de incendio (m)",
                        minimum=0,
                        maximum=8000,
                        value=1500,
                        step=50,
                        info="Distancia horizontal a puntos históricos de incendio forestal"
                    )
                    
                    wilderness_area = gr.Dropdown(
                        label="Área silvestre",
                        choices=["Rawah", "Commanche", "Cache", "Neota"],
                        value="Rawah",
                        info="Designación oficial del área silvestre protegida"
                    )
                    
                    soil_type_options = [
                        "C2702", "C2703", "C2704", "C2705", "C2706", "C2717", 
                        "C3501", "C3502", "C4201", "C4703", "C4704", "C4744", 
                        "C4758", "C5101", "C5151", "C6101", "C6102", "C6731", 
                        "C7101", "C7102", "C7103", "C7201", "C7202", "C7700", 
                        "C7701", "C7702", "C7709", "C7710", "C7745", "C7746", 
                        "C7755", "C7756", "C7757", "C7790", "C8703", "C8707", 
                        "C8708", "C8771", "C8772", "C8776"
                    ]
                    soil_type = gr.Dropdown(
                        label="Tipo de suelo",
                        choices=soil_type_options,
                        value=soil_type_options[0],
                        info="Clasificación geológica del tipo de suelo"
                    )
        
        # Panel de resultados
        with gr.Row():
            predict_button = gr.Button(" Realizar Predicción", variant="primary", size="lg")
        
        with gr.Row():
            prediction_output = gr.HTML()
    
    # Panel de información
    with gr.Tab("ℹ️ Información"):
        gr.Markdown("""
        # Acerca del Predictor de Cobertura Forestal
        
        Esta aplicación utiliza modelos de aprendizaje automático para predecir el tipo de cobertura forestal basándose en datos cartográficos.
        
        ## Tipos de Cobertura Forestal
        
        Los modelos pueden predecir siete tipos diferentes de cobertura:
        
        1. **Spruce/Fir** - Abeto/Pícea
        2. **Lodgepole Pine** - Pino contorta
        3. **Ponderosa Pine** - Pino ponderosa
        4. **Cottonwood/Willow** - Álamo/Sauce
        5. **Aspen** - Álamo temblón
        6. **Douglas-fir** - Abeto de Douglas
        7. **Krummholz** - Vegetación arbustiva alpina
        
        ## Variables predictoras
        
        Las variables utilizadas para la predicción corresponden a datos cartográficos del Servicio Forestal y de Conservación de Suelos:
        
        - **Elevación** - Altura sobre el nivel del mar
        - **Aspecto** - Orientación de la pendiente (0-360°)
        - **Pendiente** - Grado de inclinación del terreno
        - **Distancias a hidrología** - Cercanía a fuentes de agua
        - **Distancia a carreteras** - Accesibilidad
        - **Sombras de colina** - Iluminación a diferentes horas del día
        - **Distancia a puntos de incendio** - Exposición histórica a incendios
        - **Área silvestre** - Designación de área protegida
        - **Tipo de suelo** - Clasificación geológica del terreno
        
        ## Desarrollo y tecnologías
        
        Esta aplicación está desarrollada con:
        - **MLflow** - Para la gestión y despliegue de modelos
        - **Gradio** - Para la interfaz de usuario
        - **Python** - Como lenguaje de programación base
        """)
    
    # Pie de página
    gr.HTML(footer_html)
    
    predict_button.click(
        fn=predict,
        inputs=[
            model_dropdown,
            elevation,
            aspect,
            slope,
            horizontal_distance_hydrology,
            vertical_distance_hydrology,
            horizontal_distance_roadways,
            hillshade_9am,
            hillshade_noon,
            hillshade_3pm,
            horizontal_distance_fire,
            wilderness_area,
            soil_type
        ],
        outputs=[prediction_output]
    )

# Iniciar la aplicación con configuración adicional
app.launch(
    share=False,
    server_name="0.0.0.0",
    server_port=8503,
    favicon_path="https://cdn-icons-png.flaticon.com/512/2361/2361192.png"
)