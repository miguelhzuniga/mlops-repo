from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define MLflow tracking URI - update with your MLflow service URL
MLFLOW_TRACKING_URI = "http://mlflow:5000"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def download_data():
    """Download diabetes dataset"""
    import os
    import requests
    
    data_dir = '/tmp/data'
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, 'diabetes.csv')
    
    if not os.path.isfile(filepath):
        url = 'https://docs.google.com/uc?export=download&id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC'
        r = requests.get(url, allow_redirects=True, stream=True)
        with open(filepath, 'wb') as f:
            f.write(r.content)
    
    return filepath

def process_data(ti):
    """Process the downloaded data"""
    filepath = ti.xcom_pull(task_ids='download_data')
    
    # Read the CSV file
    data = pd.read_csv(filepath)
    
    # Perform basic preprocessing (example)
    # Handle missing values
    data = data.fillna(0)
    
    # Save processed data
    processed_path = '/tmp/data/processed_diabetes.csv'
    data.to_csv(processed_path, index=False)
    
    return processed_path

def train_model(ti):
    """Train a model and log to MLflow"""
    processed_path = ti.xcom_pull(task_ids='process_data')
    
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("diabetes-classification")
    
    # Load data
    data = pd.read_csv(processed_path)
    
    # Identify target variable - for this example we'll use readmitted 
    # (This may need adjustment based on the actual diabetes dataset structure)
    if 'readmitted' in data.columns:
        target = 'readmitted'
    else:
        # Fallback to a column that exists
        target = data.columns[-1]
    
    # Convert categorical variables to dummies
    categorical_columns = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    # Prepare features and target
    X = data.drop(target, axis=1)
    y = data[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Set parameters
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
        
        # Train model
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Save model
        mlflow.sklearn.log_model(clf, "random_forest_model")
        
        # Log the feature importance
        feature_importance_df = pd.DataFrame(
            data=clf.feature_importances_,
            index=X.columns,
            columns=["importance"]
        ).sort_values("importance", ascending=False)
        
        # Log feature importance as artifact
        feature_importance_path = "/tmp/feature_importance.csv"
        feature_importance_df.to_csv(feature_importance_path)
        mlflow.log_artifact(feature_importance_path)
        
        return run.info.run_id

# Define DAG
with DAG(
    'diabetes_classification',
    default_args=default_args,
    description='A DAG for diabetes classification with MLflow integration',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2025, 4, 28),
    catchup=False,
    tags=['example', 'mlops'],
) as dag:
    
    download_task = PythonOperator(
        task_id='download_data',
        python_callable=download_data,
    )
    
    process_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data,
    )
    
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )
    
    # Set task dependencies
    download_task >> process_task >> train_task