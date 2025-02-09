from joblib import load
from fastapi import FastAPI, File, UploadFile
import pandas as pd
from pydantic import BaseModel

app = FastAPI()
model, preprocessor = load("model.pkl")

print(model)

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

@app.get("/get-item/{item_id}")
def get_item(item_id: int):
    if item_id in items:
        return items[item_id]
    return {"message": "API de predicción de pingüinos activa"}

# Ruta POST
@app.post("/items/{item_id}")
def create_item(item_id: int, item: Item):
    if item_id in items:
        return {"Error": "Item exists"}
    data = pd.DataFrame([item.model_dump()])
    data = preprocessor.transform(data)
    specie = model.predict(data)[0]
    item_up = item.copy(update={"species": specie})
    items[item_id] =  item_up
    return item_up