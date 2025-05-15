import requests
import time
import random
import numpy as np
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LoadTester")

API_HOST = os.getenv("API_HOST", "ml-api-service")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}/predict"

REQUEST_INTERVAL = float(os.getenv("REQUEST_INTERVAL", "1.0"))

SEPAL_LENGTH_RANGE = (4.3, 7.9)  
SEPAL_WIDTH_RANGE = (2.0, 4.4)   
PETAL_LENGTH_RANGE = (1.0, 6.9)  
PETAL_WIDTH_RANGE = (0.1, 2.5)  

def generate_random_iris():
    """Genera características aleatorias de iris dentro de rangos realistas."""
    return {
        "sepal_length": round(random.uniform(*SEPAL_LENGTH_RANGE), 1),
        "sepal_width": round(random.uniform(*SEPAL_WIDTH_RANGE), 1),
        "petal_length": round(random.uniform(*PETAL_LENGTH_RANGE), 1),
        "petal_width": round(random.uniform(*PETAL_WIDTH_RANGE), 1)
    }

def send_request():
    """Envía una solicitud a la API de predicción de iris."""
    features = generate_random_iris()
    
    try:
        start_time = time.time()
        response = requests.post(
            API_URL,
            json=features,
            timeout=5
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            logger.info(
                f"Iris: {features} | "
                f"Especie: {result['species']} | "
                f"Probabilidad: {result['probability']:.4f} | "
                f"Tiempo API: {result['processing_time']:.4f}s | "
                f"Tiempo total: {elapsed_time:.4f}s"
            )
        else:
            logger.error(f"Error {response.status_code}: {response.text}")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error de conexión: {str(e)}")
        time.sleep(5)  
        
def main():
    """Función principal que ejecuta el bucle de carga."""
    logger.info(f"Iniciando LoadTester - Enviando solicitudes a {API_URL}")
    logger.info(f"Intervalo entre solicitudes: {REQUEST_INTERVAL} segundos")
    
    while True:
        try:
            send_request()
            jitter = random.uniform(0.8, 1.2)
            time.sleep(REQUEST_INTERVAL * jitter)
        except Exception as e:
            logger.error(f"Error inesperado: {str(e)}")
            time.sleep(REQUEST_INTERVAL)

if __name__ == "__main__":
    time.sleep(10)
    main()