import gevent
from locust import HttpUser, SequentialTaskSet, task, between

class FlujoDeInferencia(SequentialTaskSet):

    @task
    def obtener_modelos(self):
        headers = {"Connection": "close"}

        try:
            response = self.client.get("/models", headers=headers, name="📦 /models")
            if response.status_code != 200:
                print("❌ Error al obtener modelos:", response.text)
        except Exception as e:
            print(f"💥 Excepción en /models: {e}")
    @task
    def seleccionar_modelo(self):
        gevent.sleep(5)
        payload_modelo = {"model_name": "modelo1"}
        headers = {"Connection": "close"}

        try:
            response = self.client.post("/select-model", json=payload_modelo, headers=headers, name="🎯 /select-model")
            if response.status_code != 200:
                print("❌ Error al seleccionar modelo:", response.text)
            else:
                print("✅ Modelo seleccionado correctamente")
        except Exception as e:
            print(f"💥 Excepción en /select-model: {e}")
    @task
    def hacer_prediccion(self):
        gevent.sleep(5)
        payload_predict = {
            "Elevation": 1,
            "Aspect": 1,
            "Slope": 1,
            "Horizontal_Distance_To_Hydrology": 1,
            "Vertical_Distance_To_Hydrology": 1,
            "Horizontal_Distance_To_Roadways": 1,
            "Hillshade_9am": 1,
            "Hillshade_Noon": 1,
            "Hillshade_3pm": 1,
            "Horizontal_Distance_To_Fire_Points": 1,
            "Wilderness_Area": "Rawah",
            "Soil_Type": "C7745"
        }
        headers = {"Connection": "close"}

        try:
            response = self.client.post("/predict", json=payload_predict, headers=headers, name="🔮 /predict")
            if response.status_code != 200:
                print(f"❌ Error en la inferencia ({response.status_code}): {response.text}")
        except Exception as e:
            print(f"💥 Excepción en /predict: {e}")

class UsuarioDeCarga(HttpUser):
    wait_time = between(1, 5)
    tasks = [FlujoDeInferencia]
