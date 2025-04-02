from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Path to the models file
RUTA_MODELOS = "/opt/airflow/models/model.pkl"

# Load model function with error handling
def leer_modelo(nombre_archivo):
    try:
        return joblib.load(nombre_archivo)
    except Exception as e:
        print(f"Error al cargar los modelos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al cargar los modelos: {str(e)}")

items = {int: dict}

class Item(BaseModel):
    island: str
    culmen_length_mm: float
    culmen_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: str
    species: str = None

class Item2(BaseModel):
    modelo: str

@app.get("/")
def home():
    global RUTA_MODELOS
    modelos = leer_modelo(RUTA_MODELOS)
    string_modelos = ', '.join(map(str, modelos.keys()))

    return {
        "message": (
            "¡Hola, para iniciar dirígete a http://127.0.0.1:8888/docs\n"
            "Sigue los siguientes pasos:\n"
            "1. Debes asignar un número de item para cada predicción.\n"
            "2. Debes tabular los 6 campos requeridos para la predicción.\n"
            f"3. Debes seleccionar un modelo entre: {string_modelos}.\n"
            "Nota: El modelo debe ser escrito exactamente como en las opciones presentadas en el paso 3 "
            "para evitar errores al momento de predecir."
        )
    }


@app.get("/get-item/{item_id}")
def get_item(item_id: int):
    if item_id in items:
        return items[item_id]
    return {"message": "API de predicción de pingüinos activa"}


@app.post("/items/{item_id}")
def create_item(item_id: int, item: Item, modelo:Item2):
    global RUTA_MODELOS
    try:
        if item_id in items:
            return {"Error": "Item exists"}
        
        models = leer_modelo(RUTA_MODELOS)
        
        modelo_escogido = modelo.modelo
        if modelo_escogido not in models:
            return {"Error": f"Modelo '{modelo_escogido}' no encontrado. Modelos disponibles: {list(models.keys())}"}
        
        model_pipeline = models[modelo_escogido]
        
        data = pd.DataFrame([item.dict(exclude={"species"})])
        
        try:
            specie = model_pipeline.predict(data)[0]
        except Exception as e:
            return {"Error": f"Error al hacer la predicción: {str(e)}", 
                    "Detalle": f"Datos de entrada: {data.to_dict()}"}
        

        item_up = item.copy(update={"species": specie})
        
        item2_up = modelo.copy(update={"modelo": modelo_escogido})
        
        return item_up, item2_up
        
    except Exception as e:
        return {"Error": f"Error en el procesamiento: {str(e)}"}