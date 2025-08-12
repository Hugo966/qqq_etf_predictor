import requests
import pandas as pd
from io import StringIO
import os
from dotenv import load_dotenv

#Carga de variables de entorno
load_dotenv()
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

#Parámetros de configuración
SYMBOL = 'QQQ'
INTERVAL = '5min'
ANIO = 2025
FROM_MONTH = 5
TO_MONTH = 8
OUTPUT_FILE = f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_historico_5m\\QQQ_{INTERVAL}_data_{ANIO}.csv"

#Función para descargar datos intradía y fusionarlos si el CSV ya existe
def descargar_datos_intradia(anio, simbolo, intervalo, from_month, to_month, output_file, api_key):
    meses = list(range(to_month, from_month - 1, -1))  #Recorremos en orden inverso

    #Cargar CSV existente si lo hay
    if os.path.exists(output_file):
        df_existente = pd.read_csv(output_file, parse_dates=['timestamp'])
    else:
        df_existente = pd.DataFrame()

    df_total = []

    #Carga por meses
    for month in meses:
        print(f"Descargando datos para {anio}-{month:02d}...")

        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={simbolo}&interval={intervalo}&apikey={api_key}&month={anio}-{month:02d}&datatype=csv&outputsize=full'
        r = requests.get(url)

        if r.status_code == 200:
            df = pd.read_csv(StringIO(r.text))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df_month = df[(df['timestamp'].dt.year == anio) & (df['timestamp'].dt.month == month)]
            df_total.append(df_month)
        else:
            print(f"Error al descargar datos para {anio}-{month:02d}: {r.status_code}")

    #Concatenamos todos los meses descargados
    if df_total:
        df_nuevo = pd.concat(df_total, ignore_index=True)
        df_combinado = pd.concat([df_existente, df_nuevo], ignore_index=True)
        df_combinado.drop_duplicates(subset='timestamp', inplace=True)
        df_combinado = df_combinado.sort_values('timestamp')
        df_combinado.to_csv(output_file, index=False)
        print(f"Archivo actualizado en: {output_file}")
    else:
        print("No se descargaron datos nuevos.")

#Llamada a la función
descargar_datos_intradia(ANIO, SYMBOL, INTERVAL, FROM_MONTH, TO_MONTH, OUTPUT_FILE, API_KEY)
