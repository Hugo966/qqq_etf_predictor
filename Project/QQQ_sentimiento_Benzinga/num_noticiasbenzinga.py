import json

ruta_archivo = "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_sentimiento_Benzinga\\noticias_benzinga_2020.json"

#Ver número de noticias en el archivo JSON
with open(ruta_archivo, "r", encoding="utf-8") as f:
    noticias = json.load(f)

print(f"El archivo contiene {len(noticias)} noticias.")
