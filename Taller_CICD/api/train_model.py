import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os
from data_preparation import cargar_datos, preparar_datos

# Asegurarse de que existe el directorio para el modelo
os.makedirs(os.path.dirname("app/model.pkl"), exist_ok=True)

# Verificar si existe la carpeta de datos
if not os.path.exists("data"):
    os.makedirs("data")
    print("Carpeta 'data' creada. Por favor, coloca tus archivos de datos allí.")
    # Opcional: Crear datos de ejemplo si no existen
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Series(iris.target).map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    df.to_csv('data/iris.csv', index=False)
    print("Archivo de ejemplo 'data/iris.csv' creado")

# Buscar archivos CSV en la carpeta 'data'
csv_files = [f for f in os.listdir("data") if f.endswith('.csv')]
if not csv_files:
    print("No se encontraron archivos CSV en la carpeta 'data'")
    exit(1)

# Cargar el primer archivo CSV encontrado
data_file = os.path.join("data", csv_files[0])
print(f"Usando archivo de datos: {data_file}")

# Cargar y preparar datos
df = cargar_datos(data_file)
X_train, X_test, y_train, y_test, preprocessor = preparar_datos(df)


# Crear modelo KNN
print("Creando modelo KNN...")
knn = KNeighborsClassifier(n_neighbors=5)

# Crear pipeline con el preprocesador y el modelo
modelo_knn = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', knn)
])

# Entrenar modelo
print("Entrenando modelo...")
print(y_train)

modelo_knn.fit(X_train, y_train)

# Evaluar modelo
y_pred_train = modelo_knn.predict(X_train)
y_pred_test = modelo_knn.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Precisión en entrenamiento: {train_accuracy:.4f}")
print(f"Precisión en prueba: {test_accuracy:.4f}")

# Guardar modelo
print("Guardando modelo...")
joblib.dump(modelo_knn, "app/model.pkl")

print("Modelo KNN guardado como 'app/model.pkl'")

# Guardar metadatos del modelo
model_info = {
    'feature_names': list(X_train.columns) if hasattr(X_train, 'columns') else None,
    'class_names': sorted(list(set(y_train))),
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy
}

with open("app/model_info.pkl", "wb") as f:
    joblib.dump(model_info, f)

print("Información del modelo guardada como 'app/model_info.pkl'")