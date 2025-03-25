import requests
import time

#server_url = "http://server:8000/hello"
server_url = "http://api:80/data"


time.sleep(5)  # Espera a que el servidor se inicie

try:
    #response = requests.get(server_url)
    params = {"group_number": 1}

    response = requests.get(server_url, params=params)

    print("Respuesta del servidor:", response.json())
except Exception as e:
    print("Error:", e)