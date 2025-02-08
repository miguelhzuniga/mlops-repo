from data_preparation import cargar_datos, preparar_datos
from model_creation import construir_modelo, entrenar_modelo, validar_modelo

df = cargar_datos("penguins_lter.csv")

X_train, X_test, y_train, y_test, preprocessor = preparar_datos(df)

modelo = construir_modelo(preprocessor)

entrenar_modelo(modelo, X_train, y_train)

train_accuracy, test_accuracy = validar_modelo(modelo, X_test, y_test, X_train, y_train)

print(f'Accuracy en entrenamiento: {train_accuracy:.2f}')
print(f'Accuracy en prueba: {test_accuracy:.2f}')