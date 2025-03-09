from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import os
import pickle
from datetime import datetime, timedelta

# Definición de los argumentos del DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Definir el DAG
dag = DAG(
    '4-Entrenamiento_model',
    default_args=default_args,
    description='DAG para entrenar modelos de machine learning',
    schedule_interval=None,  # Solo ejecución manual si es "None"
    start_date=datetime(2025, 3, 8),
    catchup=False
)


processed_dir = '/opt/airflow/data/processed_data'
temp_dir = '/opt/airflow/data/temp'


def ensure_temp_dir():
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs('/opt/airflow/models', exist_ok=True)


def cargar_datos_modelo(**kwargs):
    ensure_temp_dir()
    
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).squeeze()  # Convertir a Serie
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).squeeze()  # Convertir a Serie

    preprocessor = joblib.load(os.path.join(processed_dir, 'preprocessor.pkl'))
    
    print("Columnas en X_train:", X_train.columns.tolist())
    
    X_train_path = os.path.join(temp_dir, 'X_train_temp.csv')
    X_test_path = os.path.join(temp_dir, 'X_test_temp.csv')
    y_train_path = os.path.join(temp_dir, 'y_train_temp.csv')
    y_test_path = os.path.join(temp_dir, 'y_test_temp.csv')
    preprocessor_path = os.path.join(temp_dir, 'preprocessor_temp.pkl')
    
    X_train.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    pd.DataFrame(y_train).to_csv(y_train_path, index=False)
    pd.DataFrame(y_test).to_csv(y_test_path, index=False)
    joblib.dump(preprocessor, preprocessor_path)
    
    return {
        'X_train_path': X_train_path,
        'X_test_path': X_test_path,
        'y_train_path': y_train_path,
        'y_test_path': y_test_path,
        'preprocessor_path': preprocessor_path,
        'columnas': X_train.columns.tolist()  
    }

def construir_modelo(**kwargs):
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='cargar_datos_modelo')
    
    knn = KNeighborsClassifier(n_neighbors=5)
    logreg = LogisticRegression(random_state=42)
    logregCV = LogisticRegressionCV(random_state=42)
    
    models = {'KNN': knn, 'LogReg': logreg, 'LogRegCV': logregCV}
    
    models_paths = {}
    for name, model in models.items():
        model_path = os.path.join(temp_dir, f'model_{name}_temp.pkl')
        joblib.dump(model, model_path)
        models_paths[name] = model_path
    
    return {
        'model_paths': models_paths,
        'columns_info': paths['columnas'] 
    }

def entrenar_modelo(**kwargs):
    ti = kwargs['ti']
    model_info = ti.xcom_pull(task_ids='construir_modelo')
    paths = ti.xcom_pull(task_ids='cargar_datos_modelo')
    
    model_paths = model_info['model_paths']
    
    X_train = pd.read_csv(paths['X_train_path'])
    y_train = pd.read_csv(paths['y_train_path']).squeeze()
    
    print("Columnas disponibles en X_train:", X_train.columns.tolist())
    
    trained_model_paths = {}
    for name, model_path in model_paths.items():
        model = joblib.load(model_path)
        model.fit(X_train, y_train)
        trained_path = os.path.join(temp_dir, f'trained_{name}_temp.pkl')
        joblib.dump(model, trained_path)
        trained_model_paths[name] = trained_path
    
    return {
        'trained_model_paths': trained_model_paths
    }

def validar_modelo(**kwargs):
    ti = kwargs['ti']
    trained_info = ti.xcom_pull(task_ids='entrenar_modelo')
    paths = ti.xcom_pull(task_ids='cargar_datos_modelo')

    trained_model_paths = trained_info['trained_model_paths']
    X_test = pd.read_csv(paths['X_test_path'])
    y_test = pd.read_csv(paths['y_test_path']).squeeze()
    X_train = pd.read_csv(paths['X_train_path'])
    y_train = pd.read_csv(paths['y_train_path']).squeeze()

    resultados = {}

    models = {}

    for nombre, model_path in trained_model_paths.items():
        modelo = joblib.load(model_path)

        y_pred_test = modelo.predict(X_test)
        y_pred_train = modelo.predict(X_train)

        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        resultados[nombre] = {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy)
        }

        # Store models in a dictionary
        models[nombre] = modelo

    mejor_modelo = max(resultados.items(), key=lambda x: x[1]['test_accuracy'])
    mejor_nombre = mejor_modelo[0]

    modelo_final = models[mejor_nombre]
    modelo_final_path = '/opt/airflow/models/model.pkl'

    # Save the models dictionary to a .pkl file
    joblib.dump(models, modelo_final_path)

    return {
        'mejor_modelo': mejor_nombre,
        'train_accuracy': resultados[mejor_nombre]['train_accuracy'],
        'test_accuracy': resultados[mejor_nombre]['test_accuracy'],
        'todos_resultados': resultados
    }


# Definir las tareas
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

cargar_datos_task >> construir_modelo_task >> entrenar_modelo_task >> validar_modelo_task