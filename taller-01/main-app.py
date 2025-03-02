from joblib import load
from fastapi import FastAPI, File, UploadFile
import pandas as pd
from pydantic import BaseModel
 
app = FastAPI()
model, preprocessor = load("model.pkl")
<<<<<<< HEAD
 
=======

<<<<<<< HEAD
>>>>>>> 53364723d38940547937c499141372a8beadfc9b
=======
>>>>>>> 89aec48b56334c813e340b0f7c4271d5a76975ab
>>>>>>> a89e3d252019f64809e0d0415f3bddf20629c45e
items = {int: dict}
 
# Definiendo un modelo de datos
class Item(BaseModel):
    island: str
    culmen_length_mm: float
    culmen_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: str
    species: None
<<<<<<< HEAD
 
class Item2(BaseModel):
    modelo: str
 
=======

class Item2(BaseModel):
    modelo: str

<<<<<<< HEAD
>>>>>>> 53364723d38940547937c499141372a8beadfc9b
=======
>>>>>>> 89aec48b56334c813e340b0f7c4271d5a76975ab
>>>>>>> a89e3d252019f64809e0d0415f3bddf20629c45e
@app.get("/")  # Definimos una ruta GET
def home():  # Definimos una función llamada home
    return {
        "message": (
            "¡Hola, para iniciar dirígete a http://127.0.0.1:8000/docs\n"
            "Sigue los siguientes pasos:\n"
            "1. Debes asignar un número de item para cada predicción.\n"
            "2. Debes tabular los 6 campos requeridos para la predicción.\n"
            "3. Debes seleccionar un modelo entre: 'KNN' y 'LogReg'.\n"
            "Nota: El modelo debe ser escrito exactamente como en las opciones presentadas en el paso 3\n"
            "para evitar errores al momento de predecir."
        )
    }  # Retornamos un diccionario con un mensaje
<<<<<<< HEAD
 
 
=======


<<<<<<< HEAD
>>>>>>> 53364723d38940547937c499141372a8beadfc9b
=======
>>>>>>> 89aec48b56334c813e340b0f7c4271d5a76975ab
>>>>>>> a89e3d252019f64809e0d0415f3bddf20629c45e
@app.get("/get-item/{item_id}")
def get_item(item_id: int):
    if item_id in items:
        return items[item_id]
    return {"message": "API de predicción de pingüinos activa"}
 
# Ruta POST
@app.post("/items/{item_id}")
def create_item(item_id: int, item: Item, modelo:Item2):
    if item_id in items:
        return {"Error": "Item exists"}
    data = pd.DataFrame([item.model_dump()])
    #data = preprocessor.transform(data)
    modelo_escogido = modelo.modelo
    if modelo_escogido not in model:
        return {"Error": f"Modelo '{modelo_escogido}' no encontrado. Modelos disponibles: {list(model.keys())}"}
    model_pipeline = model[modelo_escogido]
    specie = model_pipeline.predict(data)[0]
    item_up = item.copy(update={"species": specie})
    item2_up = modelo.copy(update={"modelo": modelo_escogido})
    items[item_id] =  item_up
<<<<<<< HEAD
 
=======

<<<<<<< HEAD
>>>>>>> 53364723d38940547937c499141372a8beadfc9b
=======
>>>>>>> 89aec48b56334c813e340b0f7c4271d5a76975ab
>>>>>>> a89e3d252019f64809e0d0415f3bddf20629c45e
    return item_up,item2_up