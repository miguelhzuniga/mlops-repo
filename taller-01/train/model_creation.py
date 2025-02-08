from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

def construir_modelo(preprocessor):

    knn = KNeighborsClassifier(n_neighbors=5)

    modelo = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('knn', knn)
    ])
    
    return modelo

def entrenar_modelo(modelo, X_train, y_train):
    modelo.fit(X_train, y_train)
    joblib.dump(modelo['knn'], r"../model.pkl")

def validar_modelo(modelo, X_test, y_test, X_train, y_train):
    y_pred_test = modelo.predict(X_test)
    y_pred_train = modelo.predict(X_train)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    return train_accuracy, test_accuracy
