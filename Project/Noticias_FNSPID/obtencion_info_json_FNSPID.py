import csv
from datetime import datetime

#Rutas
csv_input_path = "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\Noticias_FNSPID\\noticias_tickers_QQQ.csv"
csv_output_path = "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\Noticias_FNSPID\\noticias_procesadas_FNSPID.csv"

noticias_filtradas = []

#Leer CSV original
with open(csv_input_path, mode='r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)

    for row in reader:
        fecha_str = row.get("Date", "")
        titulo = row.get("Article_title", "")
        descripcion = row.get("Textrank_summary", "")

        #Limpiar saltos de línea
        descripcion_limpia = descripcion.replace("\n", " ").replace("\r", " ").strip()
        full_text = f"{titulo}. {descripcion_limpia}"

        try:
            fecha_dt = datetime.fromisoformat(fecha_str.replace("Z", "+00:00"))
            #Excluir noticias posteriores a 2024 aunque no debería haber
            if fecha_dt.year > 2024:
                continue  
        #Ignorar si no se puede parsear la fecha
        except Exception:
            continue  

        #Guardamos los datos con la fecha parseada para ordenar luego
        noticias_filtradas.append({
            "fecha": fecha_dt,
            "fecha_str": fecha_str,
            "titulo": titulo,
            "descripcion": descripcion_limpia,
            "full_text": full_text
        })

#Ordenar cronológicamente
noticias_ordenadas = sorted(noticias_filtradas, key=lambda x: x["fecha"])

#Escribir CSV de salida
with open(csv_output_path, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["Fecha", "Título", "Descripción", "Texto_completo", "Entidad", "Sentiment_score"])

    for noticia in noticias_ordenadas:
        writer.writerow([
            noticia["fecha_str"],
            noticia["titulo"],
            noticia["descripcion"],
            noticia["full_text"],
            "",
            ""
        ])

print(f"Archivo guardado correctamente en orden cronológico en: {csv_output_path}")
