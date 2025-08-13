import os
import requests
import datetime
import pandas as pd
from dotenv import load_dotenv
import json

#Configuración
load_dotenv()
API_KEY = os.getenv("MARKETAUX_API_TOKEN")
ANIO = 2021
published_before = f"{ANIO}-02-01T00:00:00"
published_after = f"{ANIO}-01-01T00:00:00"


#Petición
for i in range(1, 13):
    published_after = f"{ANIO}-{i:02d}-01T00:00:00"
    if i == 12:
        published_before = f"{ANIO + 1}-01-01T00:00:00"
    else:
        published_before = f"{ANIO}-{i + 1:02d}-01T00:00:00"
        
    url = f"https://api.marketaux.com/v1/news/all?symbols=QQQ&filter_entities=True&language=en&published_before={published_before}&published_after={published_after}&sort=relevance_score&api_token={API_KEY}"
    print("URL que se va a llamar:")
    print(url)
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()  
    print(url)

    #Recorremos los artículos y filtramos las entidades
    for article in data.get("data", []):
        #Filtrar entities: sólo mantenemos las que sean QQQ
        article["entities"] = [
            ent for ent in article.get("entities", [])
            if ent.get("symbol") == "QQQ"
        ]

    metadata = data.get("meta", [])
    df = pd.json_normalize(metadata)
    print(f"\nSe han encontrado {df["found"].values[0]} noticias")

    ruta_archivo = "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_sentimiento_Marketaux\\noticias_mes.json"

    #Leer contenido anterior si existe
    if os.path.exists(ruta_archivo):
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            datos = json.load(f)
    #Si no existe, creamos lista nueva
    else:
        datos = []

    #Obtener el nuevo valor de "found" del DataFrame
    nuevo_valor = int(df["found"].iloc[0])

    #Añadir el nuevo valor a la lista
    datos.append(nuevo_valor)

    #Guardar de nuevo
    with open(ruta_archivo, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=4)