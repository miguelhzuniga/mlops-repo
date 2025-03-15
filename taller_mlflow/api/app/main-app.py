import os
import mlflow
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

# Configuración de MLFlow
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://10.43.101.175:9000"
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'
mlflow.set_tracking_uri("http://10.43.101.175:5000")
model_name = "modelo1"
model_production_uri = f"models:/{model_name}/production"
loaded_model = mlflow.pyfunc.load_model(model_uri=model_production_uri)


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

    class Config:
        schema_extra = {
            "example": {
                "Elevation": 1000,
                "Aspect": 180,
                "Slope": 15,
                "Horizontal_Distance_To_Hydrology": 200,
                "Vertical_Distance_To_Hydrology": 100,
                "Horizontal_Distance_To_Roadways": 50,
                "Hillshade_9am": 230,
                "Hillshade_Noon": 240,
                "Hillshade_3pm": 250,
                "Horizontal_Distance_To_Fire_Points": 150,
                "Wilderness_Area": "Wilderness_Area_1",
                "Soil_Type": "Soil_Type_1"
            }
        }


app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "PRUEBA."
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
        
        input_df = pd.DataFrame(input_dict)

        prediction = loaded_model.predict(input_df)

        return {"prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))