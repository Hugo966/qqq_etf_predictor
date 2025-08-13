import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import os
import random

#Función de pérdida RMSE para la predicción del cierre, no considerando el sentimiento
def rmse_close_only(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true[:, 0] - y_pred[:, 0])))

#Fijamos semilla global
SEED = 24
os.environ['PYTHONHASHSEED'] = str(SEED) # Python hash seed
random.seed(SEED) # módulo random
np.random.seed(SEED) # NumPy

#Número de timesteps y predicciones autoregresivas
TS = 160               # Número de pasos de tiempo para la RNN
#future_steps = 240     # Número de predicciones autoregresivas a generar

#Cargamos el CSV
df = pd.read_csv(
    'C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_historico_5m\\QQQ_5min_data_2020.csv'
)

#Preprocesamos: Eliminamos timestamp y open para el conjunto de train y definimos el target
features = df[['open', 'sentiment_score', 'volume']].values
target = df[['close', 'sentiment_score', 'volume']].values

#Normalizamos los datos
data_scaler = StandardScaler()
features_scaled = data_scaler.fit_transform(features)
scaler_target = StandardScaler()
target_scaled = scaler_target.fit_transform(target) # Normalizamos el target

#Función para crear las secuencias
def create_sequences(data, target, time_steps=TS):
    X, y = [], [] #listas vacías
    #Recorremos según time_steps y creamos las ventanas tanto para las features como para el target
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(target[i + time_steps])
    #Develvemos ambas listas como arrays de numpy
    return np.array(X), np.array(y)

#Creamos las secuencias
X, y = create_sequences(features_scaled, target_scaled, time_steps=TS)

#Separamos en train, val y test sin shuffle, y separamos test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, shuffle=False
)
X_train, X_val, y_train, y_val = train_test_split(  #15% para validación
    X_temp, y_temp, test_size=0.15 / 0.85, shuffle=False  #15% de los datos originales
)

#Modelo RNN (LSTM)
#Modelo 2, con 2 capas LSTM de 50 neuronas y una capa densa final. Sin dropout final
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(TS, X.shape[2])),
    Dropout(0.1),
    LSTM(50, return_sequences=False),
    Dense(3)
])

#Compilación con optimizador Adam y función de pérdida RMSE
#RMSE FALSO PORQUE SE CALCULA LOS VALORES NORMALIZADOS, FALTA CALLBACK 
#QUE HAY EN LOS CÓDIGOS NO AUTOREGRESIVOS
#Pero después al hacer las predicciones con el otro código, se desnormalizan
model.compile(optimizer='adam', loss=rmse_close_only)

#Entrenamiento del modelo, con 100 épocas y batch_size de 32
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

#Guardamos el modelo
model.save('C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\RNNs\\modelos_y_graficas\\Ronda5_vol_sent\\modelo_2(sin_dropout)_240ts_100e_es_valloss.h5')

