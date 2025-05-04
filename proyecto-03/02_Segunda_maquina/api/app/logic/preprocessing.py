import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def preprocess_data(input_data):
    """
    Preprocesa los datos de entrada para que sean compatibles con el modelo.
    
    Args:
        input_data: Datos de entrada en formato DiabetesData (Pydantic)
        
    Returns:
        DataFrame de pandas preprocesado
    """
    logger.info("Preprocesando datos de entrada")
    
    data_dict = input_data.dict()
    
    df = pd.DataFrame([data_dict])
    
    categorical_cols = ['gender', 'age', 'max_glu_serum', 'A1Cresult', 'diabetesMed']
    
    for col in categorical_cols:
        prefix = f"{col}_"
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        
        for dummy_col in dummies.columns:
            df[dummy_col] = dummies[dummy_col].astype(int)
    
    med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                'miglitol', 'troglitazone', 'tolazamide', 'examide',
                'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
    
    df['total_diabetes_meds'] = df[med_cols].apply(lambda x: sum(x != 'No'), axis=1)
    
    if 'change' in df.columns:
        df['changed_med'] = df['change'].apply(lambda x: 1 if x == 'Ch' else 0)
    else:
        df['changed_med'] = 0  
    
    logger.info(f"Dimensiones de datos preprocesados: {df.shape}")
    return df