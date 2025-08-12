#Este código se ejecuta en env tf211
#Versión optimizada para uso mínimo de memoria en Cloud Run con TensorFlow

import asyncio
import os
from flask import Flask, request, abort
import websockets
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from google.cloud import storage
import tensorflow as tf
import sys
import pytz

#Configuramos la zona horaria de Nueva York e inicializamos la aplicación Flask
NY_TZ = pytz.timezone('America/New_York')
sys.stdout.reconfigure(line_buffering=True)
app = Flask(__name__)

#Variables globales de configuración
API_KEY = os.getenv("API_KEY_TWELVE_DATA")
SYMBOL = "QQQ"
BUCKET_NAME = "qqq-data"
DATA_FOLDER = "QQQ-data/"
MODEL_FOLDER = "modelos/"
TFLITE_MODEL_FILE = "modelo_2(no_dropout)_240ts_100e_es38_valloss3.9e-04.tflite"
TS = 240
FUTURE_STEPS = [1, 3, 6, 12, 24, 48]

#Inicializar cliente y bucket de Google Cloud Storage
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

#Función para redondear hacia abajo las marcas de tiempo a múltiplos de 5 minutos
def floor_timestamp_5min(dt):
    discard = pd.Timedelta(minutes=dt.minute % 5,
                           seconds=dt.second,
                           microseconds=dt.microsecond)
    dt_floor = dt - discard
    return dt_floor.replace(tzinfo=None)

#Función para actualizar el CSV en GCP con el nuevo precio
def update_csv_with_price(new_price, current_time):
    blob_path = f"{DATA_FOLDER}QQQ_5min_intra_market_only_ws_data.csv"
    blob = bucket.blob(blob_path)

    content = blob.download_as_bytes()
    df = pd.read_csv(pd.io.common.BytesIO(content), parse_dates=['timestamp'])

    #Convertir a hora NY y redondear hacia abajo a múltiplos de 5 min
    dt_ny = current_time.astimezone(NY_TZ)
    dt_floor = floor_timestamp_5min(dt_ny)
    timestamp_str = dt_floor.strftime('%Y-%m-%d %H:%M:%S')

    #La última ejecución diaria del código se realiza a las 15:59 ya que a las 16:00 el mercado estaría cerrado
    #El precio del activo a las 15:59 es el precio que se toma como cierre del intervalo 15:55-16:00
    if dt_ny.hour == 15 and dt_ny.minute == 59:
    #Si el minuto es 14, 29, 44 o 59
    #if dt_ny.minute in [14, 29, 44, 59]:
        if not df.empty and pd.isna(df.iloc[-1]['close']):
            df.at[df.index[-1], 'close'] = new_price #Aquí entra, porque no ejecuta esta línea?
            print(f"[15:59] Cerrando día con precio: {new_price}")
        else:
            print("[15:59] Última fila ya tenía close, no se actualiza.")

    else:
        #Caso general: cada 5 minutos normales, el precio de cierre es el precio de apertura del siguiente intervalo
        if not df.empty and pd.isna(df.iloc[-1]['close']):
            df.at[df.index[-1], 'close'] = new_price

        #Las columnas que usamos y obtenemos del WebSocket son 'timestamp', 'open' y 'close', por tanto para obtener
        #un código que se ejecute indefinidamente, solo usamos estas columnas
        new_row = {
            'timestamp': timestamp_str,
            'open': new_price,
            'close': np.nan,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    #Guardar y subir CSV actualizado
    temp_path = "/tmp/qqq_ws_data.csv"
    df.to_csv(temp_path, index=False)
    blob.upload_from_filename(temp_path)

#Función para descargar el CSV desde GCP y cargarlo en un DataFrame
def download_blob_to_df(blob_path):
    blob = bucket.blob(blob_path)
    content = blob.download_as_bytes()
    df = pd.read_csv(pd.io.common.BytesIO(content))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

#Función para descargar el modelo TFLite desde GCP
def download_tflite_model():
    
    #Creamos un directorio temporal (si no existe) para almacenar el modelo TFLite
    temp_dir = "/tmp/model"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, TFLITE_MODEL_FILE)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    blob = bucket.blob(MODEL_FOLDER + TFLITE_MODEL_FILE)
    blob.download_to_filename(temp_path)

    #Cargamos el intérprete TFLite y lo devolvemos
    interpreter = tf.lite.Interpreter(model_path=temp_path)
    interpreter.allocate_tensors()
    return interpreter

#Función para crear secuencias de datos para el modelo (por esto es necesario cargar los datos históricos)
def create_sequences(X, y, time_steps=TS):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

#Función para cargar todos los datos históricos desde el CSV en GCP 
#(como trabajo futuro, se podría optimizar para cargar solo lo necesario)
def load_all_historical_data():
    blob_path = f"{DATA_FOLDER}QQQ_5min_intra_market_only_ws_data.csv"
    df_all = download_blob_to_df(blob_path)
    df_all = df_all[~df_all.index.duplicated(keep='last')]
    return df_all

#Función para predecir el siguiente valor usando el modelo TFLite
def predict_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #Aseguramos que el tipo de dato sea float32 y la forma sea correcta
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0][0]

#Función que conecta al WebSocket de Twelve Data y realiza las predicciones
async def connect_and_predict():
    #Descargamos el modelo TFLite y los datos históricos
    interpreter = download_tflite_model()
    df = load_all_historical_data()

    #Escalamos los datos y creamos las secuencias
    scaler_in = StandardScaler()
    scaler_out = StandardScaler()
    open_scaled = scaler_in.fit_transform(df['open'].values.reshape(-1, 1))
    close_scaled = scaler_out.fit_transform(df['close'].values.reshape(-1, 1))
    X, y = create_sequences(open_scaled, close_scaled, TS)
    tcurrent = X[-1].copy()
    
    #Definimos la URL del WebSocket
    url = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={API_KEY}"
    
    #Conectamos al WebSocket y nos suscribimos al símbolo, obteniendo los datos en tiempo real
    async with websockets.connect(url) as websocket:
        subscribe_msg = {
            "action": "subscribe",
            "params": {"symbols": SYMBOL}
        }
        await websocket.send(json.dumps(subscribe_msg))

        #Se declara last_saved_time para controlar la frecuencia de actualización del CSV
        last_saved_time = 0

        #Se repite indefinidamente hasta que se obtenga un dato válido 1 vez
        while True:
            data_raw = await websocket.recv()
            data_json = json.loads(data_raw)

            #Si el mensaje es de tipo subscription, lo ignoramos
            if "action" in data_json and data_json["action"] == "subscription":
                continue
            
            #Si el mensaje contiene el precio y el timestamp, procesamos los datos
            if "price" in data_json and "timestamp" in data_json:
                price = float(data_json["price"])
                timestamp_unix = int(data_json["timestamp"])
                dt = datetime.fromtimestamp(timestamp_unix)
                
                #Determinamos si el timestamp es válido (dentro del horario de mercado)
                now = time.time()
                if now - last_saved_time >= 300:  #cada 5 minutos
                    
                    #Actualizamos el CSV en GCP con el nuevo precio y last_saved_time
                    update_csv_with_price(price, datetime.fromtimestamp(timestamp_unix, tz=pytz.utc))
                    last_saved_time = now

                    #Recargamos el CSV actualizado
                    df = download_blob_to_df(f"{DATA_FOLDER}QQQ_5min_intra_market_only_ws_data.csv")

                    #Reescalamos los datos y creamos las secuencias
                    open_scaled = scaler_in.transform(df['open'].values.reshape(-1, 1))
                    close_scaled = scaler_out.transform(df['close'].values.reshape(-1, 1))
                    X, y = create_sequences(open_scaled, close_scaled, TS)
                    tcurrent = X[-1].copy()

                    print(f"[{dt}] Dato recibido: {price}")

                    #Realizamos las predicciones para los distintos horizontes temporales
                    preds = {}
                    for fs in FUTURE_STEPS:
                        tcur = tcurrent.copy()
                        auto_preds = []
                        for _ in range(fs):
                            r = tcur.reshape(1, TS, 1)
                            next_pred = predict_tflite(interpreter, r)
                            auto_preds.append(next_pred)
                            tcur = np.append(tcur[1:], next_pred)
                        preds[fs] = scaler_out.inverse_transform([[auto_preds[-1]]])[0][0]

                    #Impir los resultados de las predicciones
                    for fs, pred_vals in preds.items():
                        if fs*5 < 60:
                            print(f"Predicción a {fs*5} min: {np.round(pred_vals, 2)}")
                        else:
                            print(f"Predicción a {fs*5//60} h: {np.round(pred_vals, 2)}")
                            

                    last_saved_time = now
                    break
                    
                    
#En la ruta /predict, se ejecuta el proceso que conecta al WebSocket y hace la predicción
@app.route('/predict', methods=['POST'])
def predict():
    #Si la petición viene de Google APIs, abortamos con 404
    user_agent = request.headers.get('User-Agent', '').lower()
    if 'apis-google' in user_agent:
        abort(404)
    
    #En caso contrario, iniciamos la conexión y predicción
    try:
        asyncio.run(connect_and_predict())
        return "Predicción realizada", 200
    except Exception as e:
        print(f"Error en predict(): {e}")
        return "Error en predicción", 500

#Ruta principal que indica que el servicio está activo
@app.route("/")
def home():
    return "Servicio activo"

#Función para iniciar el servidor Flask
def start_flask():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

#Iniciamos el servidor Flask en un hilo separado
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
