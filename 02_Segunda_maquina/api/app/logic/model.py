import os
import mlflow
from mlflow.tracking import MlflowClient
import logging
from fastapi import HTTPException


logger = logging.getLogger(__name__)

def load_production_model(tracking_uri):
    """
    Carga el modelo de producción desde MLflow.
    
    Args:
        tracking_uri: URI de conexión a MLflow
        
    Returns:
        Tuple con el modelo cargado y su información
    """
    try:
        logger.info(f"Conectando a MLflow en {tracking_uri}")
        client = MlflowClient(tracking_uri)
        
        models = client.search_registered_models()
        
        if not models:
            logger.error("No se encontraron modelos registrados en MLflow")
            raise HTTPException(status_code=500, detail="No se encontraron modelos registrados en MLflow")
        
        production_model = None
        model_name = None
        
        for model in models:
            model_name = model.name
            versions = client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                production_model = versions[0]
                break
        
        if not production_model:
            model_name = models[0].name
            versions = client.get_latest_versions(model_name)
            if versions:
                production_model = versions[0]
            else:
                logger.error(f"No se encontraron versiones para el modelo {model_name}")
                raise HTTPException(status_code=500, detail=f"No se encontraron versiones para el modelo {model_name}")
        
        logger.info(f"Cargando modelo {model_name} versión {production_model.version} (run_id: {production_model.run_id})")
        loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{production_model.version}")
        
        model_info = {
            "model_name": model_name,
            "model_version": production_model.version,
            "run_id": production_model.run_id,
            "creation_timestamp": production_model.creation_timestamp,
            "description": production_model.description
        }
        
        return loaded_model, model_info
        
    except Exception as e:
        logger.error(f"Error al cargar modelo de producción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al cargar modelo de producción: {str(e)}")