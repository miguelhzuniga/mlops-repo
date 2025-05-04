import requests
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data(ttl=300)  
def get_model_info(api_url):
    """
    Obtiene la información del modelo desde la API.
    
    Args:
        api_url: URL de la API
        
    Returns:
        Dict con información del modelo o None en caso de error
    """
    try:
        response = requests.get(f"{api_url}/model_info")
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error al obtener información del modelo: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error al conectar con la API: {str(e)}")
        return None

def predict(api_url, data):
    """
    Realiza una predicción enviando datos a la API.
    
    Args:
        api_url: URL de la API
        data: Dict con datos del paciente
        
    Returns:
        Dict con resultados de la predicción o None en caso de error
    """
    try:
        response = requests.post(f"{api_url}/predict", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error al realizar predicción: {response.status_code}")
            st.error(f"Error: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error al conectar con la API: {str(e)}")
        st.error(f"Error de conexión: {str(e)}")
        return None