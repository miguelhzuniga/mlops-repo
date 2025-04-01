import os
import mlflow
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
import time
from mlflow.exceptions import MlflowException

# Configuración de MLFlow
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://10.43.101.202:9000"
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'
mlflow.set_tracking_uri("http://10.43.101.202:5000")
model_name = "modelo1"
model_production_uri = f"models:/{model_name}/production"

def load_model_until_available(model_uri, retries=30, delay=20):

    attempt = 0
    
    while attempt < retries:
        try:
            # Try to load the model
            print(f"Attempting to load model from {model_uri} (Attempt {attempt + 1}/{retries})")
            loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
            print(f"Model loaded successfully from {model_uri}")
            return loaded_model
        except MlflowException as e:
            # If model is not available, handle exception and retry
            print(f"Model not found: {str(e)}. Retrying in {delay} seconds...")
            attempt += 1
            time.sleep(delay)  # Wait for some time before retrying
    
    # If the model isn't loaded after retries, raise an error
    raise Exception(f"Model could not be loaded after {retries} attempts.")

# Call the function to load the model
try:
    loaded_model = load_model_until_available(model_production_uri, retries=5, delay=10)
except Exception as e:
    print(f"Error: {e}")

items = {int: dict}

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

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "TALLER MLFLOW"
    }

@app.post("/predict")
async def predict(input_data: ModelInput):
    try:
        input_dict = {
            "Elevation": [input_data.Elevation],
            "Aspect": [input_data.Aspect],
            "Slope": [input_data.Slope],
            "Horizontal_Distance_To_Hydrology": [input_data.Horizontal_Distance_To_Hydrology],
            "Vertical_Distance_To_Hydrology": [input_data.Vertical_Distance_To_Hydrology],
            "Horizontal_Distance_To_Roadways": [input_data.Horizontal_Distance_To_Roadways],
            "Hillshade_9am": [input_data.Hillshade_9am],
            "Hillshade_Noon": [input_data.Hillshade_Noon],
            "Hillshade_3pm": [input_data.Hillshade_3pm],
            "Horizontal_Distance_To_Fire_Points": [input_data.Horizontal_Distance_To_Fire_Points],
            "Wilderness_Area": [input_data.Wilderness_Area],
            "Soil_Type": [input_data.Soil_Type]
        }
        
        input_df = pd.DataFrame.from_dict(input_dict, orient="columns")

        int_columns = [
            "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
            "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points"
        ]
        input_df[int_columns] = input_df[int_columns].astype("int64")

        prediction = loaded_model.predict(input_df)

        input_dict["predicción"] = prediction.tolist() if hasattr(prediction, "tolist") else prediction

        return jsonable_encoder(input_dict)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
