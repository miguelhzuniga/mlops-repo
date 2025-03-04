
import tensorflow as tf
import tensorflow_transform as tft
<<<<<<< HEAD
<<<<<<< HEAD
 
=======
<<<<<<< HEAD

>>>>>>> c2614de (basura2)
# Declaración de constantes para características numéricas y categóricas.
NUMERIC_FEATURES = [
    'Elevation', 'Aspect', 'Slope',
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
]
 
CATEGORICAL_FEATURES = [
    'Wilderness_Area', 'Soil_Type'
]
 
# Llave para la etiqueta
LABEL_KEY = 'Cover_Type'
 
def preprocessing_fn(inputs):
    outputs = {}
<<<<<<< HEAD
 
=======
=======
 
# Declaración de constantes para características numéricas y categóricas.
NUMERIC_FEATURES = [
    'Elevation', 'Aspect', 'Slope',
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
]
 
CATEGORICAL_FEATURES = [
    'Wilderness_Area', 'Soil_Type'
]
 
# Llave para la etiqueta
LABEL_KEY = 'Cover_Type'
 
def preprocessing_fn(inputs):
    outputs = {}
 
>>>>>>> af15e5b (basura)
>>>>>>> c2614de (basura2)
=======

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
>>>>>>> master
    # Transformaciones para características numéricas:
    # Las primeras tres características se escalan a un rango [0, 1].
    for key in NUMERIC_FEATURES[:3]:
        outputs[key + '_scaled_0_1'] = _fillna(tft.scale_to_0_1(
<<<<<<< HEAD
<<<<<<< HEAD
            tft.fill_missing(tf.cast(inputs[key], tf.float32))
=======
<<<<<<< HEAD
            tf.cast(inputs[key], tf.float32)
>>>>>>> c2614de (basura2)
        ))
 
    # Las siguientes tres características se escalan usando min-max scaling.
    for key in NUMERIC_FEATURES[3:6]:
        outputs[key + '_scaled_minmax'] = _fillna(tft.scale_by_min_max(
            tft.fill_missing(tf.cast(inputs[key], tf.float32))
        ))
 
    # Las restantes se escalan usando z-score normalization.
    for key in NUMERIC_FEATURES[6:]:
        outputs[key + '_scaled_zscore'] = _fillna(tft.scale_to_z_score(
            tft.fill_missing(tf.cast(inputs[key], tf.float32))
        ))
 
    # Transformaciones para características categóricas:
    for key in CATEGORICAL_FEATURES:
        # Convertir cadenas a índices de vocabulario, similar a la codificación en A.
        vocab = tft.compute_and_apply_vocabulary(
            tft.fill_missing(inputs[key])
        )
        outputs[key + '_indexed'] = vocab
        # Además, aplicar hashing para obtener una representación adicional.
        outputs[key + '_hashed'] = _fillna(tft.hash_strings(
            tft.fill_missing(inputs[key]), hash_buckets=100
        ))
 
    # La etiqueta se mantiene sin transformación.
    outputs[LABEL_KEY] = _fillna(tft.fill_missing(inputs[LABEL_KEY]))
 
    return outputs
<<<<<<< HEAD
 
=======

=======
            tft.fill_missing(tf.cast(inputs[key], tf.float32))
        ))
 
    # Las siguientes tres características se escalan usando min-max scaling.
    for key in NUMERIC_FEATURES[3:6]:
        outputs[key + '_scaled_minmax'] = _fillna(tft.scale_by_min_max(
            tft.fill_missing(tf.cast(inputs[key], tf.float32))
        ))
 
    # Las restantes se escalan usando z-score normalization.
    for key in NUMERIC_FEATURES[6:]:
        outputs[key + '_scaled_zscore'] = _fillna(tft.scale_to_z_score(
            tft.fill_missing(tf.cast(inputs[key], tf.float32))
        ))
 
    # Transformaciones para características categóricas:
    for key in CATEGORICAL_FEATURES:
        # Convertir cadenas a índices de vocabulario, similar a la codificación en A.
        vocab = tft.compute_and_apply_vocabulary(
            tft.fill_missing(inputs[key])
        )
        outputs[key + '_indexed'] = vocab
        # Además, aplicar hashing para obtener una representación adicional.
        outputs[key + '_hashed'] = _fillna(tft.hash_strings(
            tft.fill_missing(inputs[key]), hash_buckets=100
        ))
 
    # La etiqueta se mantiene sin transformación.
    outputs[LABEL_KEY] = _fillna(tft.fill_missing(inputs[LABEL_KEY]))
 
    return outputs
 
>>>>>>> af15e5b (basura)
>>>>>>> c2614de (basura2)
=======
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

>>>>>>> master
def _fillna(t, value=0):
    if not isinstance(t, tf.sparse.SparseTensor):
        return t
    return tf.squeeze(tf.sparse.to_dense(
        tf.SparseTensor(t.indices, t.values, [t.dense_shape[0], 1]),
<<<<<<< HEAD
<<<<<<< HEAD
        value), axis=1)
 
=======
<<<<<<< HEAD
        default_value=value), axis=1)


=======
        value), axis=1)
 
>>>>>>> af15e5b (basura)
>>>>>>> c2614de (basura2)
=======
        default_value=value), axis=1)


>>>>>>> master
