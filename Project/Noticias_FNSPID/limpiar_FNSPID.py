import pandas as pd
import os

ruta_entrada = "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\Noticias_FNSPID\\nasdaq_exteral_data.csv"
ruta_temporal = "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\Noticias_FNSPID\\temporal_filtrado.csv"

#Recorremos por chunks para poder cargarlo en memoria
CHUNK_SIZE = 100_000
primera_iteracion = True

#Solo queremos noticias a partir de 2020
for i, chunk in enumerate(pd.read_csv(ruta_entrada, chunksize=CHUNK_SIZE)):
    print(f"Procesando chunk {i + 1}...")

    chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')
    chunk_filtrado = chunk[chunk['Date'] >= "2020-01-01"]

    chunk_filtrado.to_csv(
        ruta_temporal,
        mode='w' if primera_iteracion else 'a',
        index=False,
        header=primera_iteracion
    )

    primera_iteracion = False

#Eliminamos el archivo original y renombramos el filtrado
os.remove(ruta_entrada)
os.rename(ruta_temporal, ruta_entrada)

print("Proceso completado. El archivo original ha sido reemplazado por la versión filtrada.")
