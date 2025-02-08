# Predicción de la Especie de Pingüinos 🐧

Este repositorio contiene un taller en el que se entrena un modelo de Machine Learning para predecir la especie de un pingüino usando el dataset **Penguins**. Además, se crea un API con **FastAPI** para realizar inferencias y se empaqueta todo en un contenedor Docker.

## 📂 Contenido del Taller

### ocesamiento y Entrenamiento del Modelo  
Se implementa un script en Python que: 
- Carga los datos del dataset. 
- Realiza la carga, limpieza, transformación, validación, ingeniería de características y división preprocesamiento, limpieza y transformación de datos. 
- Entrena un modelo de clasificación para predecir la especie del pingüino. 
- Guarda el modelo entrenado para ser usado posteriormente.

###reación de un API con FastAPI  
Se desarrolla un servicio API que permite: 
- Permite hacer inferencia al modelo entrenado. 
- Método que permite seleccionar cual modelo será usado en el proceso de inferencia.

### 4️⃣ Contenerización con Docker  
Se crea una imagen Docker para el API, exponiéndola en el puerto **8989**.
