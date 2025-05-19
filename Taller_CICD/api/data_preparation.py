import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def cargar_datos(ruta):
    df = pd.read_csv(ruta, na_values=['NA', '.'])
    print(df.dtypes)
    return df

def preparar_datos(df):
    X = df.drop(['species'], axis=1)
    y = df['species']

    # Separar columnas numéricas y categóricas
    categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
    numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

    # Pipelines de transformación
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Crear el preprocesador combinando ambos
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test, preprocessor
