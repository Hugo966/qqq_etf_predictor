import os
import requests
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import json

#Importamos y declaramos variables de entorno con valores por defecto
load_dotenv()
API_KEY = os.getenv("MARKETAUX_API_TOKEN")
BASE_URL = "https://api.marketaux.com/v1/news/all"
SYMBOL = "QQQ"
filter_entities = True
group_similar = True
language = "en" #Los modelos para analizar el sentimiento están entrenados en inglés
sort = "relevance_score"

def marketaux_call(date_from, date_to, symbol=SYMBOL, api_key=API_KEY):
    
    #Costruimos la URL con los parámetros correspondientes
    url = (
        f"{BASE_URL}"
        f"?symbols={symbol}"
        f"&filter_entities={filter_entities}"
        f"&language={language}"
        f"&published_before={date_to}"
        f"&published_after={date_from}"
        f"&sort={sort}"
        f"&api_token={API_KEY}"
    )

    #Realizamos la petición
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    #Si data no esta vacío (ha recibido 0 noticias/artículos)
    if not data.get("data"):
        print("No se han encontrado noticias/artículos para el símbolo especificado.")
    else:
        for article in data["data"]:
            #Filtrar entidades QQQ en el artículo principal
            article["entities"] = [
                ent for ent in article.get("entities", [])
                if ent.get("symbol") == "QQQ"
            ]

            #Eliminar el campo 'similar'
            article.pop("similar", None)

        
        metadata = data["meta"]
        print(f"Se han encontrado {metadata['found']} noticias/artículos")
        print(f"Se han procesado {len(data['data'])} artículos para el símbolo {symbol}")
        
        #Quitamos la metadata de cara al JSON
        data = data["data"]

        ruta_archivo = "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_sentimiento_Marketaux\\noticias_recogidas_marketaux.json"

        if os.path.exists(ruta_archivo):
            try:
                with open(ruta_archivo, "r", encoding="utf-8") as f:
                    contenido_existente = json.load(f)
            except json.JSONDecodeError:
                #Por si el archivo está vacío o corrupto
                contenido_existente = []
        else:
            contenido_existente = []

        #Añadimos los nuevos datos al contenido existente
        contenido_existente.extend(data)

        with open(ruta_archivo, "w", encoding="utf-8") as f:
            json.dump(contenido_existente, f, ensure_ascii=False, indent=4)
    
        #Prints informativos    
        df = pd.json_normalize(metadata)



published_after = "2025-04-20T00:00:00"
start_date = datetime.fromisoformat(published_after)

#Número de dias a recoger
n_calls = 100
#Calcular la fecha final
end_date = start_date + timedelta(days=n_calls)

for i in range(n_calls):
    date_from = start_date + timedelta(days=i)
    #Dia siguiente
    date_to = date_from + timedelta(days=1)
    print(date_from, date_to)
    #Convertir a string ISO
    date_from_str = date_from.isoformat()
    date_to_str = date_to.isoformat()

    #Llamada a la API con ese rango de 1 día
    marketaux_call(date_from_str, date_to_str, SYMBOL, API_KEY)

#Print informativo
print("Ultima fecha procesada:", date_from_str)