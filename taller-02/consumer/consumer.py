import time
import os

data_path = "/data"

while True:
    files = os.listdir(data_path)
    if files:
        for file in files:
            with open(os.path.join(data_path, file), "r") as f:
                print(f"\n--- Contenido de {file} ---\n")
                print(f.read())
    else:
        print("Esperando archivos...")

    time.sleep(5)