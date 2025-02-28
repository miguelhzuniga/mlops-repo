from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
 
def construir_modelo(preprocessor):
    knn = KNeighborsClassifier(n_neighbors=5)
    logreg = LogisticRegression(random_state=42)
   
    modelo_knn = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', knn)
    ])
   
    modelo_logreg = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', logreg)
    ])
   
    return {'KNN': modelo_knn, 'LogReg': modelo_logreg}
 
def entrenar_modelo(modelos, X_train, y_train):
    metricas = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        metricas[nombre] = modelo
   
    return metricas
 
def validar_modelo(modelos, X_test, y_test, X_train, y_train,preprocessor):
    resultados = {}
   
    for nombre, modelo in modelos.items():
        y_pred_test = modelo.predict(X_test)
        y_pred_train = modelo.predict(X_train)
       
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
       
        resultados[nombre] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        }
   
 
    mejor_modelo = max(resultados.items(), key=lambda x: x[1]['test_accuracy'])
<<<<<<< HEAD
   
    joblib.dump((modelos,preprocessor), r"../model.pkl")
   
=======
    
    joblib.dump((modelos,preprocessor), r"../model.pkl")
    
>>>>>>> 89aec48b56334c813e340b0f7c4271d5a76975ab
    return resultados[mejor_modelo[0]]['train_accuracy'], resultados[mejor_modelo[0]]['test_accuracy']