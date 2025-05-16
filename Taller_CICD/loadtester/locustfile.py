from locust import HttpUser, task, between
import random

SEPAL_LENGTH_RANGE = (4.3, 7.9)  
SEPAL_WIDTH_RANGE = (2.0, 4.4)   
PETAL_LENGTH_RANGE = (1.0, 6.9)  
PETAL_WIDTH_RANGE = (0.1, 2.5)   

def generate_random_iris():
    return {
        "sepal_length": round(random.uniform(*SEPAL_LENGTH_RANGE), 1),
        "sepal_width": round(random.uniform(*SEPAL_WIDTH_RANGE), 1),
        "petal_length": round(random.uniform(*PETAL_LENGTH_RANGE), 1),
        "petal_width": round(random.uniform(*PETAL_WIDTH_RANGE), 1)
    }

class LoadTester(HttpUser):
    wait_time = between(20, 60)  # Tiempo aleatorio entre solicitudes

    @task
    def predict(self):
        payload = generate_random_iris()
        self.client.post("/predict", json=payload)
