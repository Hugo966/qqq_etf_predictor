import requests
import pandas as pd
from io import StringIO
import os
from dotenv import load_dotenv

#Importamos y declaramos variables de entorno
load_dotenv()
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
SYMBOL = 'QQQ'
INTERVAL = '5min'
ANIO = 2025
OUTPUT_FILE = f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_historico_5m\\QQQ_{INTERVAL}_data_{ANIO}.csv"

# Definimos la función para descargar datos intradía
def descargar_datos_intradia(anio, simbolo, intervalo, output_file, api_key):

    #Por cada mes de un año, en orden inverso para facilitar la ordenación final del dataset resultante
    for month in range(12, 0, -1):  
        print(f"Descargando datos para {anio}-{month:02d}...")

        #Definimos la URL para la API de Alpha Vantage
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={simbolo}&interval={intervalo}&apikey={api_key}&month={anio}-{month:02d}&datatype=csv&outputsize=full'
        #Realizamos la petición
        r = requests.get(url)

        #Si la petición es exitosa, guardamos y ordenamos los datos
        if r.status_code == 200:
            df = pd.read_csv(StringIO(r.text))
            df.to_csv(output_file, mode='w' if month == 12 else 'a', header=(month == 12), index=False)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df_month = df[(df['timestamp'].dt.year == anio) & (df['timestamp'].dt.month == month)]
            #Ordenamos cronológicamente
            df_month = df_month.sort_values('timestamp')
        else:
            print(f"Error al descargar datos para {anio}-{month:02d}: {r.status_code}") 
            
    #Invertimos el CSV finalmente para obtenerlo orddenado cronológicamente
    if os.path.exists(output_file):
        df_final = pd.read_csv(output_file)
        df_final = df_final.iloc[::-1]
        df_final.to_csv(output_file, index=False)
        print(f"Archivo guardado e invertido cronológicamente en: {output_file}")

#Llamada a la función
descargar_datos_intradia(ANIO, SYMBOL, INTERVAL, OUTPUT_FILE, API_KEY)


#Este código nos permite obtener y guardar el precio del QQQ (premecado, mercado y postmercado), cada 5 minutos, del año que se especifique 
#en un archivo csv con las columnas: timestamp, open, high, low, close, volume  y se guarda en la ruta especificada.


