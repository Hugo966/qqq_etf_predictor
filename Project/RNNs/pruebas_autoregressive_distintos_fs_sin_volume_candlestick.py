import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gc
import tensorflow as tf

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
    #Carga de datos con parseo de timestamps
    ruta_csv = (
        f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\"
        f"QQQ_historico_5m\\QQQ_5min_data_{anio}.csv"
    )
    df = pd.read_csv(ruta_csv, encoding='latin1')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    #Extraemos los datos necesarios para un gráfico de velas
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    
    #Normalizamos los datos
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(o.reshape(-1,1))
    target_scaled = scaler.fit_transform(c.reshape(-1,1))
    
    #Cargamos el modelo
    model = load_model("C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\RNNs\\modelos_y_graficas\\Modelo ganador-modelo2 240TS\\modelo_2(no_dropout)_240ts_100e_es38_valloss3.9e-04.h5")
    
    #Función para crear las secuencias
    def create_sequences(X, y, time_steps=TS):
        Xs, ys = [], [] #listas vacías
        #Recorremos según time_steps y creamos las ventanas tanto para las features como para el target
        for i in range(len(X) - time_steps):
            Xs.append(X[i:i+time_steps])
            ys.append(y[i+time_steps])
        #Develvemos ambas listas como arrays de numpy
        return np.array(Xs), np.array(ys)
    
    #Creamos las secuencias
    X, y = create_sequences(features_scaled, target_scaled, TS)
    
    #Separamos en train y test. Train no se usará para entrenar, sino que se usará como el TS de la ventana de entrada
    X_train = X[:TS]
    y_train = y[:TS]
    X_test = X[TS:]
    y_test = y[TS:]
    n_preds = len(y_test)
    
    #Predicción autoregresiva con fs (future steps) instancias predecidas autoregresivamente
    n_preds = len(y_test) #Número de predicciones a realizar
    tcurrent = X_test[0].copy() #tcurrent es la ventana de entrada, contiene los TS pasos más recientes, cada uno con sus n_features (solo open en este caso)
    auto_preds = [] #Lista para almacenar las predicciones autoregresivas
    n_features = X.shape[2] #Número de features, en este caso solo open
    cont = 0 #Contador para llevar el control de las predicciones realizadas

    #Mientras que cont sea menor que el número de predicciones totales y el mínimo entre fs y el número de predicciones restantes sea mayor que 0,
    #seguimos prediciendo el mínimo entre fs y el número de predicciones restantes
    while cont < n_preds and min(fs, n_preds - (cont)) > 0: 
        for j in range(min(fs, n_preds - (cont))): #Aseguramos que no se salga del rango
            #print(cont)
            if cont%1000 == 0:
                print(cont) #Imprimimos el contador para ver el progreso
            cont += 1       
            r = tcurrent.reshape(1, TS, X.shape[2]) #r es la ventana de entrada, contiene los TS pasos más recientes, cada uno con sus n_features (solo open en este caso)
            next_open = model.predict(r, verbose=0)[0][0] #Le pasamos (1,TS,n_features) para que prediga el siguiente valor (escalado), devuelve (1,1), por eso, obtenemos el valor con [0][0]
            auto_preds.append(next_open) #Añadimos la predicción a lista de predicciones
            tcurrent = np.append(tcurrent[1:], next_open) #El nuevo tcurrent se coge desde el segundo elemento y se le añade la predicción escalada

        tcurrent = tcurrent[:-1 * min(fs, n_preds - (cont))] #Cuando hemos predicho fs pasos de manera autoregresiva, eliminamos las predicciones de tcurrent ya que añadiremos los valores reales de esos fs pasos

        if cont < len(X_test): #Si cont es menor que el tamaño de X_test, significa que hay más valores reales a agregar
            valores_reales_a_agregar = []
            for fila in X_test[cont][-min(fs, n_preds - cont):]:
                valores_reales_a_agregar.append(fila[0])
            tcurrent = np.append(tcurrent, valores_reales_a_agregar) #Añadimos los valores reales a tcurrent


    #Convertir la lista de predicciones a array y reshape para que scaler.inverse_transform pueda trabajar
    auto_preds_array = np.array(auto_preds).reshape(-1, 1)

    #Desescalamos predicciones
    auto_preds = scaler.inverse_transform(auto_preds_array).flatten()

    #Desescalamos tanto y_test como y_train para poder calcular el RMSE y visualizarlos correctamente
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    #RMSE
    mse_auto = mean_squared_error(y_test, auto_preds)
    rmse_auto = np.sqrt(mse_auto)
    print(f"RMSE autoregresivo (test completo): {rmse_auto:.4f}")
    
    #Reconstruimos los ejes de tiempo para la visualización (funciona sin la reconstrucción, pero no se por qué)
    #idx_full = df.index[TS:]               # empieza donde nace la 1ª predicción
    #idx_pred = idx_full[:len(auto_preds)]  # solo hasta n_preds
    
    
    #HTML con Plotly
    #Creamos la figura en Plotly
    fig = go.Figure()

    #Gráfico de velas con los datos reales
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
        y=auto_preds,
        mode='lines',
        name='Predicción autoregresiva',
        line=dict(color='blue', width=1)
    ))

    #Layout con rangebreaks para evitar huecos por días no laborables y fuera de mercado,
    #por ahora solo fuera de mercado y fines de semana, festivos hau que verlo
    fig.update_layout(
        title=f"Predicción autoregresiva ({anio}, fs={fs})",
        xaxis=dict(
            title="Tiempo",
            type="date",
            rangebreaks=[
                dict(bounds=[6, 1], pattern="day of week"),  #Ocultamos sábados y domingos
                dict(bounds=[16, 4], pattern="hour"),        #Ocultamos de 16:00 a 04:00
            ]
        ),
        yaxis_title="Precio",
        xaxis_rangeslider_visible=False,
        width=1000,
        height=600,
        legend=dict(x=0.01, y=0.99)
    )

    
    #Guardamos HTML
    html_path = (
        f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\RNNs\\modelos_y_graficas\\Ronda3\\modelo_ganador_{anio}_TS{TS}_fs{fs}_rmse{rmse_auto:.2f}.html"
    )
    fig.write_html(html_path)
    
    print(f"Gráfico guardado en {html_path}")
    
    
    #Limpiamos variables para liberar memoria, para evitar problemas de memoria al iterar
    vars_to_delete = [
        'df', 'features', 'target',
        'features_scaled', 'target_scaled',
        'X', 'y', 'X_train', 'y_train', 'X_test', 'y_test',
        'tcurrent', 'auto_preds_array', 'auto_preds',
        'fig', 'idx_full', 'idx_pred',
    ]
    clear_memory(vars_to_delete)


#TS se establece para cada modelo
TS = 240
anios = [2021]
fss = [3, 6, 12, 24, 48] 

for anio in anios:
    print(f"Predicciones para el año {anio} con TS={TS}:")
    for fs in fss:
        print(f"Predicción para el año {anio} con fs={fs}:")
        predecir_modelo_autoregresivo(TS, anio, fs)