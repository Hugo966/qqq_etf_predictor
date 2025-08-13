import pandas as pd
from collections import Counter

#Ruta CSV
ruta_csv = "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\Noticias_FNSPID\\noticias_tickers_QQQ.csv" 

#Tamaño de los chunks para leer en partes
CHUNK_SIZE = 100_000

#Contador de años
conteo_anual = Counter()

#Leer en chunks
for chunk in pd.read_csv(ruta_csv, chunksize=CHUNK_SIZE, usecols=['Date'], parse_dates=['Date'], low_memory=False):
    #Extraer año de la columna Date
    anios = chunk['Date'].dt.year
    conteo_anual.update(anios)

#Mostrar resultado
print("Número de noticias por año:\n")
for anio, cantidad in sorted(conteo_anual.items()):
    print(f"{anio}: {cantidad} noticias")
