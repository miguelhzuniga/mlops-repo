#if __name__=="__main__":
from joblib import load
from fastapi import FastAPI # Importamos la clase FastAPI

app = FastAPI() # Instanciamos la clase FastAPI

@app.get("/") # Definimos una ruta GET
def home(): # Definimos una función llamada home
    model = load("model.pkl")
    return {"message": f"Number of neighbors: {model.n_neighbors}"} # Retornamos un diccionario con un mensaje