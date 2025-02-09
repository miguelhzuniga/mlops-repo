from joblib import load
from fastapi import FastAPI, File, UploadFile
import pandas as pd

app = FastAPI()
model = load("model.pkl")
item_id = 0

@app.get("/") 
def home(): 
    return({"message": "Load a .csv file containing the data to make an inference."})
    
