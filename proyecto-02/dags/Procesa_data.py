from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime, timedelta
import pandas as pd
import os
import joblib  # Use joblib consistently
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=1)
}


dag = DAG(
    '3-Procesar_penguins_data',
    default_args=default_args,
    description='DAG para preprocesar datos de penguins para entrenamiento',
    schedule_interval=None,  
    start_date=datetime(2025, 3, 8),
    catchup=False
)

database_name = 'airflow_db'
raw_table = 'penguins'

processed_dir = 'opt/airflow/data/processed_data'
os.makedirs(processed_dir, exist_ok=True)

def get_data_from_db(**kwargs):
    mysql_hook = MySqlHook(mysql_conn_id='mysql_default')
    
    query = f"""
        SELECT 
            species,
            island,
            culmen_length_mm,
            culmen_depth_mm,
            flipper_length_mm,
            body_mass_g,
            sex
        FROM {database_name}.{raw_table}
    """
    
    df = mysql_hook.get_pandas_df(query)    
    df['flipper_length_mm'] = df['flipper_length_mm'].replace([float('inf'), float('-inf')], pd.NA)
    df['flipper_length_mm'] = df['flipper_length_mm'].fillna(0).astype(int)
    
    df['body_mass_g'] = df['body_mass_g'].replace([float('inf'), float('-inf')], pd.NA)
    df['body_mass_g'] = df['body_mass_g'].fillna(0).astype(int)

    df = df.replace(['NA', '.'], pd.NA)
   
    if df.empty:
        raise ValueError("No hay datos en la tabla de penguins para procesar")

    print(f"Datos obtenidos de la base de datos. Filas: {len(df)}")
    print(f"Tipos de datos: \n{df.dtypes}")
    
    kwargs['ti'].xcom_push(key='raw_data', value=df)
    print(df)
    return df


def preprocess_data(**kwargs):
    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids='get_raw_data', key='raw_data')
    
    df['culmen_length_mm'] = df['culmen_length_mm'].astype(float)
    df['culmen_depth_mm'] = df['culmen_depth_mm'].astype(float)
    df['flipper_length_mm'] = df['flipper_length_mm'].astype(int)
    df['body_mass_g'] = df['body_mass_g'].astype(int)
    df['sex'] = df['sex'].astype(str)
    df['species'] = df['species'].astype(str)
    
    if df is None:
        raise ValueError("No se encontraron datos para procesar")

    X = df.drop(['species'], axis=1)
    y = df['species']

    column_order = X.columns.tolist()
    joblib.dump(column_order, os.path.join(processed_dir, 'column_order.pkl'))
    print(f"Orden de columnas guardado: {column_order}")


    numerical_features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
    categorical_features = ['island', 'sex']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    X_train.to_csv(os.path.join(processed_dir, 'X_train_original.csv'), index=False)
    X_test.to_csv(os.path.join(processed_dir, 'X_test_original.csv'), index=False)

    preprocessor.fit(X_train)

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    pd.DataFrame(X_train_processed).to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    pd.DataFrame(X_test_processed).to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)
    
    print("Archivos CSV de datos de entrenamiento y prueba guardados.")

    preprocessor_filename = os.path.join(processed_dir, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_filename)
    print(f"Preprocesador guardado en {preprocessor_filename}")

    return preprocessor_filename


get_data_task = PythonOperator(
    task_id='get_raw_data',
    python_callable=get_data_from_db,
    dag=dag
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag
)

get_data_task >> preprocess_task