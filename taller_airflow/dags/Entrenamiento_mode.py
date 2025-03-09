from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.compose import ColumnTransformer
import os
from datetime import datetime, timedelta


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    '4-Entrenamiento_model',
    default_args=default_args,
    description='DAG para entrenar modelos de machine learning',
    schedule_interval=None,
    start_date=datetime(2025, 3, 8),
    catchup=False
)

processed_dir = '/opt/airflow/data/processed_data'
models_dir = '/opt/airflow/models'

def ensure_dirs():
    os.makedirs(models_dir, exist_ok=True)

def cargar_datos_modelo(**kwargs):
    ensure_dirs()
    
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).squeeze()

    preprocessor = joblib.load(os.path.join(processed_dir, 'preprocessor.pkl'))
    column_order = joblib.load(os.path.join(processed_dir, 'column_order.pkl'))
    
    print(f"Columnas en X_train: {X_train.shape[1]} columnas")
    print(f"Orden de columnas original: {column_order}")
    return {
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
        'column_order': column_order,

    }

def construir_modelo(**kwargs):
    ti = kwargs['ti']
    _ = ti.xcom_pull(task_ids='cargar_datos_modelo')
    preprocessor = joblib.load(os.path.join(processed_dir, 'preprocessor.pkl'))

    knn = KNeighborsClassifier(n_neighbors=5)
    logreg = LogisticRegression(random_state=42)
    logregCV = LogisticRegressionCV(random_state=42)

    knn_pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', knn)
    ])
    
    logreg_pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', logreg)
    ])
    
    logregCV_pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', logregCV)
    ])
    
    return {
        'models': {
            'KNN': 'KNeighborsClassifier(n_neighbors=5)',
            'LogReg': 'LogisticRegression(random_state=42)',
            'LogRegCV': 'LogisticRegressionCV(random_state=42)'
        }
    }

def entrenar_modelo(**kwargs):
    ti = kwargs['ti']
    _ = ti.xcom_pull(task_ids='construir_modelo')
    
    # Load original data directly from disk for training
    X_train_original = pd.read_csv(os.path.join(processed_dir, 'X_train_original.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).squeeze()
    preprocessor = joblib.load(os.path.join(processed_dir, 'preprocessor.pkl'))
    
    print(f"Entrenando modelos con datos originales: {X_train_original.shape}")
    
    # Create and train models directly
    models = {}
    trained_info = {}
    
    # KNN
    knn_pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', KNeighborsClassifier(n_neighbors=5))
    ])
    knn_pipeline.fit(X_train_original, y_train)
    models['KNN'] = knn_pipeline
    trained_info['KNN'] = 'trained'
    
    # LogReg
    logreg_pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', LogisticRegression(random_state=42))
    ])
    logreg_pipeline.fit(X_train_original, y_train)
    models['LogReg'] = logreg_pipeline
    trained_info['LogReg'] = 'trained'
    
    # LogRegCV
    logregCV_pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', LogisticRegressionCV(random_state=42))
    ])
    logregCV_pipeline.fit(X_train_original, y_train)
    models['LogRegCV'] = logregCV_pipeline
    trained_info['LogRegCV'] = 'trained'
    
    # Store all models in a single file
    modelo_final_path = os.path.join(models_dir, 'model.pkl')
    joblib.dump(models, modelo_final_path)
    
    return {
        'trained_models': trained_info,
        'model_path': modelo_final_path
    }

def validar_modelo(**kwargs):
    ti = kwargs['ti']
    trained_info = ti.xcom_pull(task_ids='entrenar_modelo')
    model_path = trained_info['model_path']
    
    models = joblib.load(model_path)
    
    X_test_original = pd.read_csv(os.path.join(processed_dir, 'X_test_original.csv'))
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).squeeze()
    

    X_train_original = pd.read_csv(os.path.join(processed_dir, 'X_train_original.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).squeeze()

    resultados = {}
    

    for nombre, pipeline in models.items():
        print(f"Validando modelo: {nombre}")
        y_pred_test = pipeline.predict(X_test_original)
        y_pred_train = pipeline.predict(X_train_original)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        resultados[nombre] = {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy)
        }

    mejor_modelo = max(resultados.items(), key=lambda x: x[1]['test_accuracy'])
    mejor_nombre = mejor_modelo[0]
    
    print(f"Mejor modelo: {mejor_nombre}")
    print(f"Precisión en prueba: {resultados[mejor_nombre]['test_accuracy']}")
    

    return {
        'mejor_modelo': mejor_nombre,
        'train_accuracy': resultados[mejor_nombre]['train_accuracy'],
        'test_accuracy': resultados[mejor_nombre]['test_accuracy'],
        'todos_resultados': resultados,
        'model_path': model_path
    }

def test_api_format(**kwargs):
    ti = kwargs['ti']
    validation_info = ti.xcom_pull(task_ids='validar_modelo')
    model_path = validation_info['model_path']
    try:
        models = joblib.load(model_path)
        print(f"Modelos cargados correctamente: {list(models.keys())}")
        sample_api_input = pd.DataFrame([{
            'island': 'Torgersen',
            'culmen_length_mm': 39.1,
            'culmen_depth_mm': 18.7,
            'flipper_length_mm': 181.0,
            'body_mass_g': 3750.0,
            'sex': 'Male'
        }])
        
        print(f"Entrada de prueba API: \n{sample_api_input}")
        
        for nombre, model in models.items():
            try:
                prediction = model.predict(sample_api_input)
                print(f"Modelo {nombre} predice: {prediction}")
            except Exception as e:
                print(f"Error con modelo {nombre}: {str(e)}")
    except Exception as e:
        print(f"Error al cargar los modelos: {str(e)}")
    
    return {
        'test_result': 'completed'
    }


cargar_datos_task = PythonOperator(
    task_id='cargar_datos_modelo',
    python_callable=cargar_datos_modelo,
    provide_context=True,
    dag=dag
)

construir_modelo_task = PythonOperator(
    task_id='construir_modelo',
    python_callable=construir_modelo,
    provide_context=True,
    dag=dag
)

entrenar_modelo_task = PythonOperator(
    task_id='entrenar_modelo',
    python_callable=entrenar_modelo,
    provide_context=True,
    dag=dag
)

validar_modelo_task = PythonOperator(
    task_id='validar_modelo',
    python_callable=validar_modelo,
    provide_context=True,
    dag=dag
)

test_api_task = PythonOperator(
    task_id='test_api_format',
    python_callable=test_api_format,
    provide_context=True,
    dag=dag
)

cargar_datos_task >> construir_modelo_task >> entrenar_modelo_task >> validar_modelo_task >> test_api_task