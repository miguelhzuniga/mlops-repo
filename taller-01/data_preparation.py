import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def cargar_datos(ruta):
    df = pd.read_csv(ruta, na_values=['NA', '.'])
    return df

def preparar_datos(df):
    X = df.drop(['Species'], axis=1)
    y = df['Species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('normalization', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder())
    ])

    categorical_with_missing = [col for col in df.columns if df[col].isna().any() and df[col].dtypes == object]
    numerical_with_missing = [col for col in df.columns if df[col].isna().any() and df[col].dtypes == np.float64]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_with_missing),
            ('cat', categorical_transformer, categorical_with_missing)
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor
