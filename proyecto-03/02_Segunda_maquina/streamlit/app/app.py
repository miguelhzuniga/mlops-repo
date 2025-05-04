import streamlit as st
import pandas as pd
import requests
import json
import time
import os
from streamlit.app.utils.api_client import get_model_info, predict


API_URL = os.getenv("API_URL", "http://fastapi-service:8000")


st.set_page_config(
    page_title="Predictor de Readmisión para Diabetes",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 0.5rem;
        font-weight: bold;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .positive {
        background-color: #ffcccc;
        border: 1px solid #ff8888;
    }
    .negative {
        background-color: #ccffcc;
        border: 1px solid #88ff88;
    }
    .model-info {
        font-size: 0.8rem;
        color: #666666;
        margin-top: 1rem;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 Predictor de Readmisión para Diabetes")
st.subheader("Predicción de readmisión hospitalaria para pacientes con diabetes")

with st.sidebar:
    st.header("Información del Paciente")
    st.markdown("Introduce los detalles del paciente para predecir el riesgo de readmisión")
    
    st.subheader("Demografía")
    gender = st.selectbox("Género", ["Male", "Female", "Unknown"])
    age = st.selectbox("Grupo de Edad", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", 
                                      "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])
    
    st.subheader("Información Hospitalaria")
    time_in_hospital = st.slider("Tiempo en Hospital (días)", 1, 14, 7)
    num_lab_procedures = st.slider("Número de Procedimientos de Laboratorio", 1, 120, 45)
    num_procedures = st.slider("Número de Procedimientos", 0, 6, 1)
    num_medications = st.slider("Número de Medicamentos", 1, 81, 18)
    
    st.subheader("Historial Médico Previo")
    number_outpatient = st.slider("Número de Visitas Ambulatorias", 0, 42, 0)
    number_emergency = st.slider("Número de Visitas a Emergencias", 0, 76, 0)
    number_inpatient = st.slider("Número de Hospitalizaciones", 0, 21, 1)
    number_diagnoses = st.slider("Número de Diagnósticos", 1, 16, 9)
    
    st.subheader("Resultados de Laboratorio")
    max_glu_serum = st.selectbox("Glucosa Sérica Máxima", ["None", "Norm", ">200", ">300"])
    a1c_result = st.selectbox("Resultado A1C", ["None", "Norm", ">7", ">8"])
    
    with st.expander("Diagnósticos (Opcional)"):
        diag_1 = st.text_input("Diagnóstico Primario", "428")
        diag_2 = st.text_input("Diagnóstico Secundario", "428")
        diag_3 = st.text_input("Diagnóstico Adicional", "250")
    
    with st.expander("Medicamentos (Opcional)"):
        diabetes_med = st.selectbox("Medicación para Diabetes", ["Yes", "No"])
        insulin = st.selectbox("Insulina", ["No", "Up", "Down", "Steady"])
        
        metformin = st.selectbox("Metformina", ["No", "Up", "Down", "Steady"])

    predict_btn = st.button("Predecir Riesgo de Readmisión")

col1, col2 = st.columns([3, 2])

with col1:
    st.header("Realizar una Predicción")
    
    if predict_btn:
        with st.spinner("Calculando riesgo de readmisión..."):
            data = {
                "gender": gender,
                "age": age,
                "time_in_hospital": time_in_hospital,
                "num_lab_procedures": num_lab_procedures,
                "num_procedures": num_procedures,
                "num_medications": num_medications,
                "number_outpatient": number_outpatient,
                "number_emergency": number_emergency,
                "number_inpatient": number_inpatient,
                "number_diagnoses": number_diagnoses,
                "max_glu_serum": max_glu_serum,
                "A1Cresult": a1c_result,
                "diabetesMed": diabetes_med,
                "diag_1": diag_1,
                "diag_2": diag_2,
                "diag_3": diag_3,
                "insulin": insulin,
                "metformin": metformin
            }
            
            result = predict(API_URL, data)
            
        if result:
            st.subheader("Resultado de la Predicción")
            
            readmission_status = result.get("readmission_status", "Desconocido")
            css_class = "positive" if readmission_status == "No Readmitido" else "negative"
            
            st.markdown(f"""
            <div class="result-box {css_class}">
                <h3>{readmission_status}</h3>
                <p>El modelo predice que el paciente {'no ' if readmission_status == 'No Readmitido' else ''}será readmitido dentro de 30 días.</p>
            </div>
            """, unsafe_allow_html=True)
            
            model_info = result.get("model_info", {})
            if model_info:
                st.markdown(f"""
                <div class="model-info">
                    <p><strong>Modelo:</strong> {model_info.get('model_name', 'Desconocido')}</p>
                    <p><strong>Versión:</strong> {model_info.get('model_version', 'Desconocida')}</p>
                    <p><strong>ID de Ejecución:</strong> {model_info.get('run_id', 'Desconocido')}</p>
                </div>
                """, unsafe_allow_html=True)
                
            proc_time = result.get("processing_time_ms", 0)
            st.text(f"Tiempo de procesamiento: {proc_time} ms")

with col2:
    st.header("Información del Modelo")
    
    model_info = get_model_info(API_URL)
    
    if model_info:
        st.markdown(f"""
        #### Modelo de Producción Actual
        - **Nombre:** {model_info.get('model_name', 'Desconocido')}
        - **Versión:** {model_info.get('model_version', 'Desconocida')}
        - **ID de Ejecución:** {model_info.get('run_id', 'Desconocido')}
        """)
        
        st.markdown("""
        #### Acerca de este Modelo
        Este modelo predice si un paciente diabético será readmitido al hospital dentro de 30 días basándose en su historial médico, demografía y resultados de laboratorio.
        
        #### Factores Clave que Afectan la Readmisión
        - Duración de la estancia hospitalaria
        - Número de hospitalizaciones previas
        - Número de diagnósticos
        - Número de medicamentos
        - Cantidad de procedimientos de laboratorio
        """)
    else:
        st.error("No se pudo recuperar la información del modelo. Comprueba si la API está funcionando.")

st.markdown("""
---
### Acerca de la Predicción de Readmisión por Diabetes

Esta aplicación utiliza un modelo de aprendizaje automático entrenado con datos de pacientes diabéticos de 130 hospitales de EE.UU. entre 1999-2008. El modelo evalúa el riesgo de que un paciente sea readmitido dentro de los 30 días después del alta.

### Cómo Utilizar
1. Introduce la información del paciente en la barra lateral
2. Haz clic en "Predecir Riesgo de Readmisión"
3. Revisa la predicción y los detalles del modelo

El modelo utiliza automáticamente la última versión de producción de MLflow sin requerir cambios en el código cuando se registran nuevos modelos.
""")

st.markdown("""
---
📊 **Proyecto MLOps** | Creado con Streamlit
""")