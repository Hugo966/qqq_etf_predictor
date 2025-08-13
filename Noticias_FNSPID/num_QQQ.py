import pandas as pd

ruta_entrada = "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\Noticias_FNSPID\\nasdaq_exteral_data.csv"
ruta_salida = "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\Noticias_FNSPID\\noticias_tickers_QQQ.csv"

CHUNK_SIZE = 100_000
nasdaq_tickers = {"QQQ", "TQQQ", "SQQQ"}
primera_iteracion = True

#Para cada chunk buscamos las noticias de los tickers QQQ, TQQQ y SQQQ (QQQ y apalancados)
for i, chunk in enumerate(pd.read_csv(ruta_entrada, chunksize=CHUNK_SIZE, low_memory=False)):
    print(f"Procesando chunk {i+1}...")

    #Nos aseguramos de que la columna sea string
    chunk['Stock_symbol'] = chunk['Stock_symbol'].astype(str)

    #Filtramos por tickers exactos
    chunk_filtrado = chunk[chunk['Stock_symbol'].str.upper().isin(nasdaq_tickers)]

    #Guardamos en archivo
    chunk_filtrado.to_csv(
        ruta_salida,
        mode='w' if primera_iteracion else 'a',
        index=False,
        header=primera_iteracion
    )

    primera_iteracion = False

print(f"\nProceso completado. Noticias de QQQ/TQQQ/SQQQ guardadas en:\n{ruta_salida}")
