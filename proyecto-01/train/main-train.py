if __name__=="__main__":

    from data_preparation import *
    from model_creation import *
    import os
    import requests
    ## download the dataset
    # Directory of the raw data files
    _data_root = './data/covertype'
    # Path to the raw training data
    _data_filepath = os.path.join(_data_root, 'covertype_train.csv')

    # Download data
    os.makedirs(_data_root, exist_ok=True)
    if not os.path.isfile(_data_filepath):
        #https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/
        url = 'https://docs.google.com/uc?export= \
        download&confirm={{VALUE}}&id=1lVF1BCWLH4eXXV_YOJzjR7xZjj-wAGj9'
        r = requests.get(url, allow_redirects=True, stream=True)
        open(_data_filepath, 'wb').write(r.content)
        
        df = cargar_datos(_data_filepath)

    X_train, X_test, y_train, y_test, preprocessor = preparar_datos(df)

    modelo = construir_modelo(preprocessor)

    entrenar_modelo(modelo, X_train, y_train)

    train_accuracy, test_accuracy = validar_modelo(modelo, X_test, y_test, X_train, y_train, preprocessor)

    print(f'Model created successfully')