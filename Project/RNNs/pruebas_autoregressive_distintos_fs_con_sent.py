import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import plotly.graph_objects as go
import gc

def rmse_close_only(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true[:, 0] - y_pred[:, 0])))

def clear_memory(var_names: list):
    #Eliminamos las variables del espacio de nombres global (las variables que se pasan como argumento)
    for name in var_names:
        if name in globals():
            try:
                del globals()[name]
            except:
                pass

    #Limpiamos el estado de Keras
    tf.keras.backend.clear_session()

    #Forzamos la recolección de basura para liberar memoria
    gc.collect()

def predecir_modelo_autoregresivo(TS, anio, fs):
    #Carga de datos y parseo de timestamps
    ruta_csv = f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_historico_5m\\QQQ_5min_data_{anio}.csv"
    df = pd.read_csv(ruta_csv, encoding='latin1')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    #En este caso features lo usaremos para predecir con el modelo ya entrenado, y target para comparar con las predicciones
    features = df[['open', 'sentiment_score']].values
    target = df[['close', 'sentiment_score']].values

    #Normalizamos los datos
    data_scaler = StandardScaler()
    features_scaled = data_scaler.fit_transform(features)
    scaler_target = StandardScaler()
    target_scaled = scaler_target.fit_transform(target) # Normalizamos el target

    #Cargamos el modelo con la función rmse_close_only como métrica personalizada
    model = load_model("C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\RNNs\\modelos_y_graficas\\Ronda4_sentimiento\\NUEVO2modelo_2(2+dense(20))_240ts_100e_es_valloss.h5", custom_objects={'rmse_close_only': rmse_close_only})
    
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

    #Separamos en train y test. Train no se usará para entrenar, sino que se usará como el TS de la ventana de entrada
    X_train = X[:TS]
    y_train = y[:TS]
    X_test = X[TS:]
    y_test = y[TS:]

    #Predicción autoregresiva con fs (future steps) instancias predecidas autoregresivamente
    n_preds = len(y_test) #Número de predicciones a realizar
    tcurrent = X_test[0].copy() #tcurrent es la ventana de entrada, contiene los TS pasos más recientes, cada uno con sus n_features (open y sentiment en este caso),
                                 #cada elemento de tcurrent es una tupla (open, sentiment)
    auto_preds = [] #Lista para almacenar las predicciones autoregresivas
    n_features = X.shape[2] #Número de features, open y sentiment en este caso
    cont = 0 #Contador para llevar el control de las predicciones realizadas
    #Mientras que cont sea menor que el número de predicciones totales y el mínimo entre fs y el número de predicciones restantes sea mayor que 0,
    #seguimos prediciendo el mínimo entre fs y el número de predicciones restantes
    while cont < n_preds and min(fs, n_preds - (cont)) > 0: 
        for j in range(min(fs, n_preds - (cont))): #Aseguramos que no se salga del rango
            if cont % 1000 == 0:
                print(cont) #Imprimimos el contador para ver el progreso
            cont += 1       
            r = tcurrent.reshape(1, TS, X.shape[2]) #r es la ventana de entrada, contiene los TS pasos más recientes, cada uno con sus n_features (open y sentiment en este caso)
            next_open = model.predict(r, verbose=0)[0] #Le pasamos (1,TS,n_features) para que prediga el siguiente valor (escalado), devuelve tupla de dos valores, es decir, (1,2)
            auto_preds.append(tuple(next_open)) #Añadimos la tupla de predicciones a lista de predicciones
            tcurrent = np.vstack([tcurrent[1:], next_open]) #El nuevo tcurrent se coge desde el segundo elemento y se le añade la predicción escalada
            
        #Cuando hemos predicho fs pasos de manera autoregresiva, eliminamos las predicciones de tcurrent ya que añadiremos los valores reales de esos fs pasos
        tcurrent = tcurrent[:(-1 * min(fs, n_preds-(cont)))] #Cuando hemos predicho fs pasos de manera autoregresiva, 
                                                              #eliminamos las predicciones de tcurrent ya que añadiremos los valores reales de esos fs pasos
        
        if cont < len(X_test): #Si cont es menor que el tamaño de X_test, significa que hay más valores reales a agregar
            valores_reales_a_agregar = []
            valores_reales_a_agregar = X_test[cont][-min(fs, n_preds - cont):] 
            tcurrent = np.vstack([tcurrent, valores_reales_a_agregar]) #Añadimos los valores reales a tcurrent
    
    #Convertir la lista de predicciones a array y reshape para que scaler.inverse_transform pueda trabajar
    auto_preds_array = np.array(auto_preds) #Shape final será (n_preds, 2), donde 2 son las dos features (open y sentiment en este caso)
    
    #Desescalamos predicciones
    auto_preds = scaler_target.inverse_transform(auto_preds_array)
        
    y_test = scaler_target.inverse_transform(y_test)    
    y_train = scaler_target.inverse_transform(y_train)   
    
    rmse_close = np.sqrt(mean_squared_error(y_test[:, 0], auto_preds[:, 0]))
    rmse_sent = np.sqrt(mean_squared_error(y_test[:, 1], auto_preds[:, 1]))
    
    print(f"RMSE 'close': {rmse_close:.4f}")
    print(f"RMSE 'sentiment': {rmse_sent:.4f}")

    #Reconstruimos los ejes de tiempo para la visualización
    #total_len = len(target)
    #time_idx = np.arange(total_len)
    #train_idx = time_idx[TS : TS + len(y_train)]
    #test_idx  = time_idx[TS + len(y_train) : TS + len(y_train) + len(y_test)]
    #y_preds = time_idx[TS + len(y_train) : TS + len(y_train) + len(auto_preds)]
    train_idx = df.index[TS : TS + len(y_train)]
    test_idx  = df.index[TS + len(y_train) : TS + len(y_train) + len(y_test)]
    y_preds   = df.index[TS + len(y_train) : TS + len(y_train) + len(auto_preds)]

    #Creamos la figura
    fig = go.Figure()

    #Gráfico de velas
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Precio real"
    ))

    #Gráfico de líneas con la predicción autoregresiva
    fig.add_trace(go.Scatter(
        x=df.index[TS + len(y_train) : TS + len(y_train) + len(auto_preds)],
        y=auto_preds[:, 0],  # Solo cierre
        mode='lines',
        name='Predicción autorregresiva',
        line=dict(color='blue', width=1)
    ))

    #Gráfico
    fig.update_layout(
        title=f"Predicción autorregresiva ({anio}, fs={fs})",
        xaxis=dict(
            title="Tiempo",
            type="date",
            rangebreaks=[
                dict(bounds=[6, 1], pattern="day of week"),  # Oculta sábados y domingos
                dict(bounds=[16, 4], pattern="hour")         # Oculta fuera de horas de mercado
            ]
        ),
        yaxis_title="Precio",
        xaxis_rangeslider_visible=False,
        width=1000,
        height=600,
        legend=dict(x=0.01, y=0.99)
    )

    #Guardamos la figura en un archivo HTML
    html_path = (
            f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\RNNs\\modelos_y_graficas\\Ronda4_sentimiento\\N2modelo_3(2+dense(20))_240ts_100e_es39_valloss0,0124_{anio}_fs{fs}_rmse{rmse_close:.2f}.html"
    )
    fig.write_html(html_path)
    
    
    #Sentimiento real vs predicho
    fig = go.Figure()

    #Serie de entrenamiento (real) - sentimiento
    fig.add_trace(
        go.Scatter(x=train_idx, y=y_train[:, 1], mode='lines', name='Train (real)', line=dict(width=1, color='blue'))
    )

    #Serie de test (real) - sentimiento
    fig.add_trace(
        go.Scatter(x=test_idx, y=y_test[:, 1], mode='lines', name='Test (real)', line=dict(width=1, color='orange'))
    )

    #Predicción autoregresiva - sentimiento
    fig.add_trace(
        go.Scatter(x=y_preds, y=auto_preds[:, 1], mode='lines', name='Predicción autorregresiva', line=dict(width=1, color='green'))
    )

    #Diseño del gráfico
    fig.update_layout(
        title=f"Sentimiento real vs Predicción autorregresiva año {anio} y fs={fs}",
        xaxis_title="Tiempo",
        yaxis_title="Sentiment score",
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            type="date",
            rangebreaks=[
                dict(bounds=[6, 1], pattern="day of week"),  # Oculta sábados y domingos
                dict(bounds=[16, 4], pattern="hour")         # Oculta fuera de horas de mercado
            ]
        ),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        width=900,
        height=450,
    )

    #Guardamos la figura en un archivo HTML
    html_path_sentimiento = (
        f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\RNNs\\modelos_y_graficas\\Ronda4_sentimiento\\N2SENTIMENTmodelo_3(2+dense(20))_240ts_100e_es39_valloss0,0124_{anio}_fs{fs}_rmse{rmse_close:.2f}.html"
    )
    fig.write_html(html_path_sentimiento)


    #Limpiamos variables para liberar memoria, para evitar problemas de memoria al iterar
    vars_to_delete = [
        'df', 'features', 'target',
        'features_scaled', 'target_scaled',
        'X', 'y', 'X_train', 'y_train', 'X_test', 'y_test',
        'tcurrent', 'auto_preds_array', 'auto_preds',
        'fig', 'fig2', 'train_idx', 'test_idx', 'y_preds_idx',
    ]
    clear_memory(vars_to_delete)


TS = 240
anios = [2021]  # Años a predecir
fss = [3, 6, 12, 24, 48]  

for anio in anios:
    print(f"Predicciones para el año {anio} con TS={TS}:")
    for fs in fss:
        print(f"Predicción para el año {anio} con fs={fs}:")
        predecir_modelo_autoregresivo(TS, anio, fs)
    