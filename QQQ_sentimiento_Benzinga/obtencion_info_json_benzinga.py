import json
import csv
import re

#Rutas
json_path = "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_sentimiento_Benzinga\\noticias_benzinga_2024.json" #Entrada
csv_path = "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_sentimiento_Benzinga\\noticias_procesadas_benzinga_2024.csv" #Salida

#Cargar el archivo JSON
with open(json_path, 'r', encoding='utf-8') as file:
    news_data = json.load(file)

#Abrimos el archivo CSV modo escritura
with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)

    #Añadimos la cabecera del CSV
    writer.writerow([
        "Fecha", "Título", "Descripción", "Texto_completo", "Entidad", "Sentiment_score"
    ])

    #Recorremos cada noticia
    for news in news_data:
        title = news.get("title", "")
        
        #Limpieza de saltos de línea y espacios extra del cuerpo
        body_raw = news.get("body", "")
        description = re.sub(r'\s+', ' ', body_raw).strip()
        
        #Combinar título y descripción
        full_text = f"{title}. {description}"
        time = news.get("created", "")

        #Entidad y sentimiento vacíos porque Benzinga no los proporciona
        writer.writerow([time, title, description, full_text, "", ""])

#Mensaje final
print(f"Archivo guardado correctamente en: {csv_path}")
