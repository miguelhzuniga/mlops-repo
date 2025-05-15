from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

def crear_modelo_knn(preprocessor, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    modelo_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', knn)
    ])
    
    return modelo_pipeline