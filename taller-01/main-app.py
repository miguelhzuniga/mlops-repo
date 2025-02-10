from joblib import load
from fastapi import FastAPI, File, UploadFile
import pandas as pd
from pydantic import BaseModel

app = FastAPI()
model, preprocessor = load("model.pkl")

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

class Item2(BaseModel):
    modelo: str
 
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

    return item_up,item2_up