import tensorflow as tf
import tensorflow_transform as tft

# Constantes para características
NUMERIC_FEATURES = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
]
CATEGORICAL_FEATURES = ['Wilderness_Area', 'Soil_Type']
LABEL_KEY = 'Cover_Type'

def preprocessing_fn(inputs):
    
    outputs = {}

    # Escalar características numéricas asegurando que sean float32 y sin valores nulos
    for key in NUMERIC_FEATURES[:3]:
        outputs[key + '_scaled_0_1'] = tft.scale_to_0_1(
            tft.fill_missing(tf.cast(inputs[key], tf.float32))
        )

    for key in NUMERIC_FEATURES[3:6]:
        outputs[key + '_scaled_minmax'] = tft.scale_by_min_max(
            tft.fill_missing(tf.cast(inputs[key], tf.float32))
        )

    for key in NUMERIC_FEATURES[6:]:
        outputs[key + '_scaled_zscore'] = tft.scale_to_z_score(
            tft.fill_missing(tf.cast(inputs[key], tf.float32))
        )

    # Procesar características categóricas
    for key in CATEGORICAL_FEATURES:
        # Convertir strings a índices de vocabulario
        outputs[key + '_indexed'] = tft.compute_and_apply_vocabulary(
            tft.fill_missing(inputs[key])
        )
        
        # Alternativamente, usar hash para características categóricas con muchos valores únicos
        outputs[key + '_hashed'] = tft.hash_strings(
            tft.fill_missing(inputs[key]), hash_buckets=100
        )

    # Pasar la etiqueta sin transformar (puedes normalizarla si es necesario)
    outputs[LABEL_KEY] = tft.fill_missing(inputs[LABEL_KEY])

    return outputs