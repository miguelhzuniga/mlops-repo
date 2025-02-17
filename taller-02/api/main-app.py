from joblib import load
from fastapi import FastAPI, File, UploadFile
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

items = {int: dict}

RUTA_MODELOS = "/train/model.pkl"

def leer_modelo(nombre_archivo):
    models = load(nombre_archivo)
    return models

# Definiendo un modelo de datos
class Item(BaseModel):
    island: str
    culmen_length_mm: float
    culmen_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: str
    species: None

class Item2(BaseModel):
    modelo: str

@app.get("/")  # Definimos una ruta GET
def home():  # Definimos una función llamada home
    global RUTA_MODELOS
    modelos = leer_modelo(RUTA_MODELOS)
    string_modelos = ', '.join(map(str, modelos.keys()))

    return {
        "message": (
            "¡Hola, para iniciar dirígete a http://127.0.0.1:8000/docs"
            "Sigue los siguientes pasos:"
            "1. Debes asignar un número de item para cada predicción."
            "2. Debes tabular los 6 campos requeridos para la predicción."
            f"3. Debes seleccionar un modelo entre: {string_modelos}."
            "Nota: El modelo debe ser escrito exactamente como en las opciones presentadas en el paso 3"
            "para evitar errores al momento de predecir."
        )
    }  # Retornamos un diccionario con un mensaje


@app.get("/get-item/{item_id}")
def get_item(item_id: int):
    if item_id in items:
        return items[item_id]
    return {"message": "API de predicción de pingüinos activa"}

# Ruta POST
@app.post("/items/{item_id}")
def create_item(item_id: int, item: Item, modelo:Item2):
    global RUTA_MODELOS
    model = leer_modelo(RUTA_MODELOS)
    if item_id in items:
        return {"Error": "Item exists"}
    data = pd.DataFrame([item.model_dump()])
    modelo_escogido = modelo.modelo
    if modelo_escogido not in model:
        return {"Error": f"Modelo '{modelo_escogido}' no encontrado. Modelos disponibles: {list(model.keys())}"}
    model_pipeline = model[modelo_escogido]
    specie = model_pipeline.predict(data)[0]
    item_up = item.copy(update={"species": specie})
    item2_up = modelo.copy(update={"modelo": modelo_escogido})
    items[item_id] =  item_up

    return item_up,item2_up