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
        payload_predict = {
                            "gender": "string",
                            "age": "string",
                            "time_in_hospital": 0,
                            "num_lab_procedures": 0,
                            "num_procedures": 0,
                            "num_medications": 0,
                            "number_outpatient": 0,
                            "number_emergency": 0,
                            "number_inpatient": 0,
                            "number_diagnoses": 0,
                            "max_glu_serum": "string",
                            "A1Cresult": "string",
                            "diabetesMed": "string",
                            "diag_1": "",
                            "diag_2": "",
                            "diag_3": "",
                            "metformin": "",
                            "repaglinide": "",
                            "nateglinide": "",
                            "chlorpropamide": "",
                            "glimepiride": "",
                            "acetohexamide": "",
                            "glipizide": "",
                            "glyburide": "",
                            "tolbutamide": "",
                            "pioglitazone": "",
                            "rosiglitazone": "",
                            "acarbose": "",
                            "miglitol": "",
                            "troglitazone": "",
                            "tolazamide": "",
                            "examide": "",
                            "citoglipton": "",
                            "insulin": ""
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