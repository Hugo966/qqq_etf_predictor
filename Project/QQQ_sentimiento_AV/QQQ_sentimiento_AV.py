import os
import requests
import pandas as pd
from dotenv import load_dotenv
import json

#Importamos y declaramos variables de entorno
load_dotenv()
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"
SYMBOL = "QQQ"
FECHA_INICIO = "20220101T0001"
FECHA_FIN = "20221201T0001"

#Temas de interés de cara al QQQ
topic_prueba = "technology"
#Señales de tecnología vs flujos de mercado
topics_tech_market_flow = "technology,financial_markets"
#Sentimiento de tecnología frente a tipos de interés
topics_tech_rate_sentiment = "technology,economy_monetary"
#Tendencias consumo en tecnológicas
topics_tech_retail_trends = "technology,retail_wholesale"
#Flujos de consumo vs mercado
topics_retail_market_flow = "retail_wholesale,financial_markets"

#Costruimos la URL con los parámetros correspondientes
url = (
    f"{BASE_URL}"
    f"?function=NEWS_SENTIMENT"
    f"&symbols={SYMBOL}"
    f"&time_from={FECHA_INICIO}"
    f"&time_to={FECHA_FIN}"
    f"&sort=RELEVANCE"
    f"&limit=1000"
    f"&topics={topic_prueba}"
    f"&apikey={API_KEY}"
)

#Print indicativo para realmente ver la URL que se va a llamar
print("URL que se va a llamar:")
print(url)
#URL de prueba que funciona
#url = "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&symbols=QQQ&sort=LATEST&limit=100&topics=technology&apikey=CZT41JLX3I5K8HA4"
#URL de prueba a modificar al gusto, experimental
url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&symbols=QQQ&time_from={FECHA_INICIO}&time_to={FECHA_FIN}&sort=LATEST&limit=1000&topics=technology&apikey=CZT41JLX3I5K8HA4"

#Realizamos la petición
response = requests.get(url)
response.raise_for_status()
data = response.json()

print(data)

#Guardamos la información en un archivo JSON
with open("C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_sentimiento_AV\\202404.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

#Convertimos la respuesta JSON a un DataFrame para mejor manejabilidad
feed = data.get("feed", [])
df = pd.json_normalize(data["feed"])

#Selección de columnas de interés
cols = [
    "title",
    "time_published",
    "summary",
    "topics",
    "overall_sentiment_score"
]
df = df[cols]

#Convertimos "time_published" a formato datetime
df["time_published"] = pd.to_datetime(df["time_published"], format="%Y%m%dT%H%M%S")

#Filtramos para obtener solamente el tema y su puntuación de relevancia
df["topics"] = df["topics"].apply(
    lambda lst: [
        {"topic": t.get("topic"), "relevance_score": t.get("relevance_score")}
        for t in lst
    ]
)

#Prints informativos
print(f"Se han recogido {len(df)} noticias")
print(df.head())