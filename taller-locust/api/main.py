import os
import mlflow
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from mlflow.exceptions import MlflowException
from typing import Dict, List, Optional, Union

# Configuración de MLFlow
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'
mlflow.set_tracking_uri("http://mlflow:5000")

# Variable para almacenar los modelos cargados
loaded_models = {}
current_model_name = None

class ModelInput(BaseModel):
    Elevation: int
    Aspect: int
    Slope: int
    Horizontal_Distance_To_Hydrology: int
    Vertical_Distance_To_Hydrology: int
    Horizontal_Distance_To_Roadways: int
    Hillshade_9am: int
    Hillshade_Noon: int
    Hillshade_3pm: int
    Horizontal_Distance_To_Fire_Points: int
    Wilderness_Area: str
    Soil_Type: str
    model_name: Optional[str] = None

class ModelSelection(BaseModel):
    model_name: str

app = FastAPI()

# Iniciar la aplicación sin cargar ningún modelo
print("Iniciando servicio. Los modelos se cargarán cuando el usuario los seleccione.")

@app.get("/")
def home():
    with open("index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/models")
async def list_production_models():
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
            
            # Filtrar solo las versiones en producción
            for version in client.search_model_versions(f"name='{model.name}'"):
                if version.current_stage == "Production":
                    model_info["versions"].append({
                        "version": version.version,
                        "creation_timestamp": version.creation_timestamp
                    })
            
            if model_info["versions"]:  # Agregar solo si tiene versiones en producción
                production_models.append(model_info)
        
        return {"success": True, "models": production_models}
    
    except Exception as e:
        print(f"Error al listar modelos en producción: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener modelos: {str(e)}")

@app.post("/select-model")
async def select_model(selection: ModelSelection):
    global current_model_name
    
    try:
        model_name = selection.model_name
        
        # Verificar si el modelo ya está cargado
        if model_name in loaded_models:
            current_model_name = model_name
            return {"success": True, "message": f"Modelo '{model_name}' seleccionado"}
        
        # Cargar el modelo desde MLflow (un solo intento)
        try:
            model_uri = f"models:/{model_name}/production"
            print(f"Cargando modelo desde {model_uri}")
            loaded_models[model_name] = mlflow.pyfunc.load_model(model_uri=model_uri)
            current_model_name = model_name
            print(f"Modelo '{model_name}' cargado correctamente")
            return {"success": True, "message": f"Modelo '{model_name}' cargado y seleccionado correctamente"}
        except MlflowException as e:
            print(f"Error al cargar el modelo '{model_name}': {e}")
            raise HTTPException(
                status_code=404, 
                detail=f"No se pudo cargar el modelo '{model_name}'. Verifique que exista y tenga una versión en producción."
            )
    except Exception as e:
        print(f"Error al seleccionar modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Error al seleccionar modelo: {str(e)}")

@app.post("/predict")
async def predict(input_data: ModelInput):
    # Verificar que haya un modelo seleccionado
    if not current_model_name:
        raise HTTPException(
            status_code=400, 
            detail="No hay un modelo seleccionado. Por favor, seleccione un modelo antes de realizar predicciones."
        )
    
    # Usar el modelo especificado o el actual
    model_name = input_data.model_name if input_data.model_name else current_model_name
    
    if model_name not in loaded_models:
        raise HTTPException(
            status_code=400,
            detail=f"El modelo '{model_name}' no está cargado. Por favor, selecciónelo primero."
        )
    
    try:
        # Obtener los datos en un formato que el modelo pueda usar
        input_dict = jsonable_encoder(input_data)
        del input_dict["model_name"]  # Eliminar campo no necesario para la predicción
        
        # Convertir a dataframe para la predicción
        input_df = pd.DataFrame([input_dict])
        
        # Realizar la predicción
        prediction = loaded_models[model_name].predict(input_df)
        
        return {"predicción": prediction.tolist()}
    except Exception as e:
        print(f"Error al realizar predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")