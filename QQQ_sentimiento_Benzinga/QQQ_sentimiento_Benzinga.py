import requests
import json
import os
import html
import re
from datetime import datetime, timedelta

def limpiar_html(texto_html):
    texto_sin_tags = re.sub(r'<[^>]+>', '', texto_html)
    texto_limpio = html.unescape(texto_sin_tags)
    return texto_limpio.strip()

#Configuración
token = "???"
ticker = "QQQ"
date_from = datetime.strptime("2024-01-01", "%Y-%m-%d") 
date_to = datetime.strptime("2024-12-31", "%Y-%m-%d")
ruta_archivo = "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_sentimiento_Benzinga\\noticias_benzinga_2024.json"

#Cargar contenido existente
if os.path.exists(ruta_archivo):
    try:
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            contenido_existente = json.load(f)
    except json.JSONDecodeError:
        contenido_existente = []
else:
    contenido_existente = []

#Iterar día a día
fecha_actual = date_from
while fecha_actual <= date_to:
    fecha_str = fecha_actual.strftime("%Y-%m-%d")
    print(f"📅 Buscando noticias para el día: {fecha_str}")

    url = "https://api.benzinga.com/api/v2/news"
    querystring = {
        "token": token,
        "pageSize": "25",
        "displayOutput": "full",
        "dateFrom": fecha_str,
        "dateTo": fecha_str,
        "tickers": ticker,
        "sort": "created:asc"
    }
    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code != 200:
        print(f"Error en la solicitud para el {fecha_str}: {response.status_code}")
        fecha_actual += timedelta(days=1)
        continue

    noticias = response.json()
    noticias_filtradas = []

    #Guardamos la información de interés de cada noticia
    for noticia in noticias:
        if "body" in noticia:
            noticia["body"] = limpiar_html(noticia["body"])
        if "image" in noticia:
            del noticia["image"]
        if "channels" in noticia:
            noticia["channels"] = [ch for ch in noticia["channels"] if ch.get("name") == "QQQ"]
        if "stocks" in noticia:
            noticia["stocks"] = [st for st in noticia["stocks"] if st.get("name") == "QQQ"]
        if "tags" in noticia:
            if noticia["tags"] and isinstance(noticia["tags"][0], str):
                noticia["tags"] = [tag for tag in noticia["tags"] if tag == "QQQ"]
            elif noticia["tags"] and isinstance(noticia["tags"][0], dict):
                noticia["tags"] = [tag for tag in noticia["tags"] if tag.get("name") == "QQQ"]

        if noticia.get("channels") or noticia.get("stocks") or noticia.get("tags"):
            noticias_filtradas.append(noticia)

    if noticias_filtradas:
        contenido_existente.extend(noticias_filtradas)
        print(f"{len(noticias_filtradas)} noticias guardadas para el {fecha_str}")

    #Avanza un día
    fecha_actual += timedelta(days=1)

#Guardar todo
with open(ruta_archivo, "w", encoding="utf-8") as f:
    json.dump(contenido_existente, f, ensure_ascii=False, indent=4)

print(f"\nTotal noticias acumuladas: {len(contenido_existente)}")
print(f"Archivo guardado en: {ruta_archivo}")
