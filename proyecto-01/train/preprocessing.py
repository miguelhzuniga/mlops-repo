
import tensorflow as tf
import tensorflow_transform as tft

# Declaración de constantes para características numéricas y categóricas.
NUMERIC_FEATURES = [
    'Elevation', 'Slope',
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon',
    'Horizontal_Distance_To_Fire_Points'
]
# Llave para la etiqueta
LABEL_KEY = 'Cover_Type'

def preprocessing_fn(inputs):
    outputs = {}
    # Transformaciones para características numéricas:
    # Las primeras tres características se escalan a un rango [0, 1].
    for key in NUMERIC_FEATURES[:3]:
        outputs[key + '_scaled_0_1'] = _fillna(tft.scale_to_0_1(
            tf.cast(inputs[key], tf.float32)
        ))
    # Las siguientes tres características se escalan usando min-max scaling.
    for key in NUMERIC_FEATURES[3:6]:
        outputs[key + '_scaled_minmax'] = _fillna(tft.scale_by_min_max(
            tf.cast(inputs[key], tf.float32)
        ))
    # Las restantes se escalan usando z-score normalization.
    for key in NUMERIC_FEATURES[6:]:
        outputs[key + '_scaled_zscore'] = _fillna(tft.scale_to_z_score(
            tf.cast(inputs[key], tf.float32)
        ))
    
    # La etiqueta se mantiene sin transformación.
    outputs[LABEL_KEY] = inputs[LABEL_KEY]
    return outputs

def _fillna(t, value=0):
    if not isinstance(t, tf.sparse.SparseTensor):
        return t
    return tf.squeeze(tf.sparse.to_dense(
        tf.SparseTensor(t.indices, t.values, [t.dense_shape[0], 1]),
        default_value=value), axis=1)


