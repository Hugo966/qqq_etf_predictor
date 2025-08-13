import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import random

#Fijamos semilla global
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED) # Python hash seed
random.seed(SEED) # módulo random
np.random.seed(SEED) # NumPy

#Número de timesteps y predicciones autoregresivas
TS = 240        # Número de pasos de tiempo para la RNN
#future_steps = 3     # Número de predicciones autoregresivas a generar

#Cargamos el CSV
df = pd.read_csv(
    'C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_historico_5m\\QQQ_5min_solo_ws_2020-2024.csv'
)

#Preprocesamos: Eliminamos timestamp y open para el conjunto de train y definimos el target
features = df['open'].values 
target = df['close'].values

#Normalizamos los datos
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.reshape(-1,1)) # Normalizamos las features
target_scaled = scaler.fit_transform(target.reshape(-1,1)) # Normalizamos el target

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

#Separamos en train, val y test sin shuffle, y separamos test aunque test no se use en este caso, 
#por mantener el entrenamiento y validación igual y justo para todos los modelos
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, shuffle=False
)
X_train, X_val, y_train, y_val = train_test_split(  #15% para validación
    X_temp, y_temp, test_size=0.15 / 0.85, shuffle=False  #15% de los datos originales
)

#Modelo RNN (LSTM)
#Modelo con 50 unidades LSTM, sin dropout final y dense de 20 unidades ( esto último sólo en modelo 3)
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(TS, X.shape[2])),
    Dropout(0.1),
    LSTM(50, return_sequences=False),
    #Dense(20, activation='relu'),
    Dense(1)
])

#Compilación con optimizador Adam y función de pérdida MSE
model.compile(optimizer='adam', loss='mean_squared_error')

#Entrenamiento del modelo, con 100 épocas y batch_size de 32
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

#Guardamos el modelo
model.save('C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\RNNs\\modelos_y_graficas\\Ronda2\\B1\\modelo_2(no_dropout)_240ts_2020-2025_100e_es_vallossv2.h5')

