if __name__=="__main__":

    from data_preparation import *
    from model_creation import *

    df = cargar_datos(r"data/penguins_size.csv")

    X_train, X_test, y_train, y_test, preprocessor = preparar_datos(df)

    modelo = construir_modelo(preprocessor)

    entrenar_modelo(modelo, X_train, y_train)

    train_accuracy, test_accuracy = validar_modelo(modelo, X_test, y_test, X_train, y_train, preprocessor)

    print(f'Model created successfully')