import tensorflow as tf

#Rutas tanto del modelo Keras a convertir como del modelo TFLite resultante
keras_model_path =  "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\implementación_gcp\\modelo_2(no_dropout)_240ts_100e_es38_valloss3.9e-04.keras"
tflite_model_path = "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\implementación_gcp\\modelo_2(no_dropout)_240ts_100e_es38_valloss3.9e-04.tflite"

#Carga del modelo Keras
model = tf.keras.models.load_model(keras_model_path, compile=False)

#Creación del convertidor TFLite a partir del modelo Keras
converter = tf.lite.TFLiteConverter.from_keras_model(model)

#Permitir operaciones TFLite estándar y selectas de TensorFlow
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,   #Operaciones nativas TFLite
    tf.lite.OpsSet.SELECT_TF_OPS      #Operaciones selectas TF necesarias para LSTM
]       

#Deshabilitar lowering de TensorList ops para evitar error
converter._experimental_lower_tensor_list_ops = False

#Conversión a TFLite
tflite_model = converter.convert()

#Guardado del modelo TFLite en la ruta especificada
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Modelo TFLite guardado en: {tflite_model_path}")
