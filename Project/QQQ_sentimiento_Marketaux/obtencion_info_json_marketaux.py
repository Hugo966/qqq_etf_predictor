import json
import csv
from datetime import datetime

#Rutas
json_path = "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_sentimiento_Marketaux\\noticias_recogidas_marketaux.json" #Entrada
csv_path = "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_sentimiento_Marketaux\\noticias_procesadas_marketaux.csv" #Salida

#Cargar el archivo JSON
with open(json_path, 'r', encoding='utf-8') as file:
    news_data = json.load(file)

#Abrimos el archivo CSV para escribir
with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)

    #Escribimos la cabecera del CSV
    writer.writerow([
        "Fecha", "Título", "Descripción", "Texto_completo", "Entidad", "Sentiment_score"
    ])

    #Recorremos cada noticia
    for news in news_data:
        title = news.get("title", "")
        description = news.get("description", "")
        time = news.get("published_at", "")

        #Intentamos parsear la fecha y filtramos
        try:
            fecha_dt = datetime.fromisoformat(time.replace("Z", "+00:00"))
            #ignorar noticias de 2025 en adelante
            if fecha_dt.year > 2024:
                continue  
        #Si la fecha está mal formada, la ignoramos
        except Exception:
            continue  

        #Limpiar saltos de línea y combinar el texto completo
        clean_description = description.replace("\n", " ").replace("\r", " ").strip()
        full_text = f"{title}. {clean_description}"

        #Revisamos si hay entidades con sentimiento
        entities = news.get("entities", [])
        if entities:
            for entity in entities:
                symbol = entity.get("symbol", "")
                sentiment_score = entity.get("sentiment_score", "")
                writer.writerow([time, title, clean_description, full_text, symbol, sentiment_score])
        else:
            writer.writerow([time, title, clean_description, full_text, "", ""])

print(f"Archivo guardado correctamente en: {csv_path}")
