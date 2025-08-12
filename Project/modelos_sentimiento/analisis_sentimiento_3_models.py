import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict
import numpy as np
import csv

#Creamos la lista para almacenar los resultados que escribiremos en el CSV
resultados = []

#Cargamos modelos y tokenizers
#Model: ahmedrachid/FinancialBERT-Sentiment-Analysis, negative/neutral/positive y confidence score
model_name_fba = "ahmedrachid/FinancialBERT-Sentiment-Analysis"
tokenizer_fba = AutoTokenizer.from_pretrained(model_name_fba)
model_fba = AutoModelForSequenceClassification.from_pretrained(model_name_fba)

#Model: LHF/finbert-regressor, [-1,1] -1:max_negative, 1:max_positive
model_name_fbr = "LHF/finbert-regressor"    
tokenizer_fbr = AutoTokenizer.from_pretrained("LHF/finbert-regressor")
model_fbr = AutoModelForSequenceClassification.from_pretrained("LHF/finbert-regressor")

#Model: tabularisai/robust-sentiment-analysis, Very Negative/Negative/Neutral/Positive/Very Positive y confidence score
model_name_rsa = "tabularisai/robust-sentiment-analysis"
tokenizer_rsa = AutoTokenizer.from_pretrained("tabularisai/robust-sentiment-analysis")
model_rsa = AutoModelForSequenceClassification.from_pretrained("tabularisai/robust-sentiment-analysis")

#Crear los pipelines de análisis de sentimiento
classifier_fba = pipeline("text-classification", model=model_fba, tokenizer=tokenizer_fba)
classifier_fbr = pipeline("text-classification", model=model_fbr, tokenizer=tokenizer_fbr)
classifier_rsa = pipeline("text-classification", model=model_rsa, tokenizer=tokenizer_rsa)

#Cargamos el CSV de las noticias procesadas
csv_path_noticias = "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\modelos_sentimiento\\noticias_procesadas_2020-2024.csv"
df_noticias = pd.read_csv(csv_path_noticias)

df_noticias["Fecha"] = pd.to_datetime(df_noticias["Fecha"], errors='coerce', utc=True)
df_noticias["Fecha"] = df_noticias["Fecha"].dt.tz_convert('America/New_York').dt.tz_localize(None)

#Filtramos por año correspondinte o sin año, aquí especificamos el año que queremos analizar
anio = 2024 #Poner 0 si es sin año
if anio == 0:
    #Si no se especifica año, no filtramos
    df_filtrado = df_noticias
else:
    df_filtrado = df_noticias[df_noticias['Fecha'].dt.year == anio]


#Analizar cada noticia (columna 'Texto_completo') 
#'Texto completo' = "Título. Descripción"
cont = 1
for i, row in df_filtrado.iterrows():
    texto = str(row['Texto_completo'])

    #Pasamos los textos a los modelos capando a 512 tokens
    resultado_fba = classifier_fba(texto[:512])[0]  
    resultado_fbr = classifier_fbr(texto[:512])[0]  
    resultado_rsa = classifier_rsa(texto[:512])[0]  

    #Mostrar resultados no normalizados
    print(f"Noticia {cont}:")
    print("Texto:", texto)
    print("Fecha:", row['Fecha'])
    print("Resultados no normalizados:")
    print("Predicción FinancialBERT Analysis:", resultado_fba['label'], "| Score:", round(resultado_fba['score'], 4))
    print("Predicción FinBERT Regressor:", "Score:", round(resultado_fbr['score'], 4))
    print("Predicción Robust Sentiment Analysis:", resultado_rsa['label'], "| Score:", round(resultado_rsa['score'], 4))
    print("Predicción de Marketaux:", row['Sentiment_score'])
    
    #Escalado de predicciones
    
    #ahmedrachid/FinancialBERT-Sentiment-Analysis
    if resultado_fba['label'] == 'positive':
        score_fba_norm = 0.5 + (resultado_fba['score'] / 2)
    elif resultado_fba['label'] == 'negative':
        score_fba_norm = 0.5 - (resultado_fba['score'] / 2)
    else:
        score_fba_norm = 0.5  # Neutral, confidence no sirve, no sabemos hacia que lado se inclina
    
    #LHF/finbert-regressor
    score_fbr_norm = (resultado_fbr['score'] + 1) / 2
        
    #tabularisai/robust-sentiment-analysis
    if resultado_rsa['label'] == 'Very Positive':
        score_rsa_norm = 0.9 + 0.1 * resultado_rsa['score']  # rango 0.9 a 1.0
    elif resultado_rsa['label'] == 'Positive':
        score_rsa_norm = 0.7 + 0.2 * resultado_rsa['score']  # rango 0.7 a 0.9
    elif resultado_rsa['label'] == 'Neutral':
        score_rsa_norm = 0.5
    elif resultado_rsa['label'] == 'Negative':
        score_rsa_norm = 0.3 - 0.2 * resultado_rsa['score']  # rango 0.1 a 0.3
    elif resultado_rsa['label'] == 'Very Negative':
        score_rsa_norm = 0 + 0.1 * resultado_rsa['score']    # rango 0 a 0.1
    else:
        score_rsa_norm = 0.5  # fallback neutral
        
    #Sentiment score de Marketaux
    if row['Sentiment_score'] >= 0:
        score_marketaux_norm = row['Sentiment_score'] / 2 + 0.5
    else:
        score_marketaux_norm = 0.5 + (row['Sentiment_score'] / 2)
    
    media_huggingface = (score_fba_norm + score_fbr_norm + score_rsa_norm) / 3
    #Si existe sentiment_score de Marketaux, le damos más peso
    #Si no, media_huggingface ya tiene el peso de Marketaux
    if pd.notna(row['Sentiment_score']):
        media = (media_huggingface + score_marketaux_norm) / 2
    else:
        media = media_huggingface

    
    #Mostrar resultados normalizados    
    print("\nResultados normalizados:")
    print("Predicción media escalada FinancialBERT Analysis:", round(score_fba_norm, 2))
    print("Predicción media escalada FinBERT Regressor:", round(score_fbr_norm, 2))
    print("Predicción media escalada Robust Sentiment Analysis:", round(score_rsa_norm, 2))
    print("Predicción media escalada de Marketaux:", round(score_marketaux_norm, 2))
    print("\nPredicción media modelos Hugging Face:", round(media_huggingface, 2))
    print("Predicción media con peso añadido a Marketaux:", round(media, 2), "\n")
    
    #Guardamos la predicción media directamente en el DataFrame original para poder analizarlo después
    df_filtrado.at[i, "Prediccion_media"] = round(media, 2)


    #Cambiar la zona horaria de UTC (noticias de Marketaux) a ET (timestamp de Alpha Vantage)
    #Se supone que ya las meto todas en ET?
    #fecha_utc = datetime.fromisoformat(row["Fecha"].replace("Z", "+00:00"))
    #fecha_et = fecha_utc.astimezone(ZoneInfo("America/New_York"))

    #fecha_et = datetime.fromisoformat(row["Fecha"])  # asumiendo row["Fecha"] ya con ET
    fecha_et = row["Fecha"]  # ya es datetime o Timestamp

    #Guardamos los resultados en la lista formateando la fecha
    resultados.append({
        "Fecha": fecha_et.strftime("%Y-%m-%d %H:%M:%S"),
        "Prediccion_media": round(media, 2)
    })
    
    cont += 1

#Convertir la lista de resultados en una lista de diccionarios con fechas como claves (un timestamp puede tener varias noticias) 
#noticias = resultados
noticias = [{"ts": pd.to_datetime(r["Fecha"]), "sent": r["Prediccion_media"]} for r in resultados]

#Cargar CSV original con los timestamps
csv_path = f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_historico_5m\\QQQ_5min_data_{anio}.csv"
df = pd.read_csv(csv_path)
#Asegurarse de que la columna 'timestamp' esté en formato datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])
    
#Cada noticia se asigna al intervalo de tiempo siguiente al timestamp de la noticia
market_ts = df["timestamp"].values  #array de datetime64 con timestamps del CSV
#Diccionario cuyo valor por defecto es una lista vacía
#Para acumular en cada clave (timestamp) todas las noticias asignadas a ese intervalo
news_map = defaultdict(list)

for noticia in noticias:
    #Busca la posición en market_ts donde insertar 'noticia["ts"]' de forma ordenada, el índice más pequeño pero con ts tras la noticia
    idx = market_ts.searchsorted(np.datetime64(noticia["ts"]))
    if idx < len(market_ts):
        #Si el índice está dentro del rango, añadimos la noticia al timestamp correspondiente
        news_map[market_ts[idx]].append(noticia["sent"])
    #si idx == len(market_ts), la noticia es posterior al último tick del CSV: se ignora


#Decay para sentimiento cuando no hay noticias
decay_factor = decay_factor = 0.991239 #En 12 intervalos de 5 minutos (1 hora) se reduce al 10% el sentimiento de la última noticia
sentimiento_acumulado = 0.5 #Sentimiento inicial
alpha = 0.85
sentiment_scores = []

for ts in market_ts:
    # ¿Hay noticias agrupadas en este intervalo?
    lst = news_map.get(ts, [])
    if lst:
        media_noticias = sum(lst) / len(lst)
        #Si hay noticia se aplica una EMA (Exponential Moving Average), se le da más peso a la noticia reciente
        sentimiento_acumulado = sentimiento_acumulado * (1 - alpha) + media_noticias * alpha
    else:
        #Si no hay noticia, se aplica el decay al sentimiento acumulado
        sentimiento_acumulado = sentimiento_acumulado * decay_factor + (1 - decay_factor) * 0.5

    #Capamos a 5 decimales
    sentimiento_acumulado = round(sentimiento_acumulado, 5)
    #Añadimos el sentimiento acumulado a la lista
    sentiment_scores.append(sentimiento_acumulado)

    
#Añadir la columna de sentimiento acumulado al DataFrame de precios
df["sentiment_score"] = sentiment_scores

#Copiar el resultado de la predicción media al DataFrame completo de noticias,
#sustituyendo la columna 'Sentiment_score'
df_noticias.loc[df_filtrado.index, "Sentiment_score"] = df_filtrado["Prediccion_media"]

#Guardar CSVs
df.to_csv(f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_historico_5m\\QQQ_5min_data_{anio}.csv", index=False, encoding='utf-8')
df_noticias.to_csv(csv_path_noticias, index=False, encoding='utf-8')
print("CSV actualizado con los resultados guardado correctamente.")
