import pandas as pd
from zoneinfo import ZoneInfo
from dateutil import parser

# Lista con las rutas de los CSVs a unir
csv_files = [
    "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_sentimiento_Marketaux\\noticias_procesadas_marketaux.csv",
    "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_sentimiento_Benzinga\\noticias_procesadas_benzinga_2020.csv",
    "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_sentimiento_Benzinga\\noticias_procesadas_benzinga_2021.csv",
    "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_sentimiento_Benzinga\\noticias_procesadas_benzinga_2022.csv",
    "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_sentimiento_Benzinga\\noticias_procesadas_benzinga_2023.csv",
    "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_sentimiento_Benzinga\\noticias_procesadas_benzinga_2024.csv",
    "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\Noticias_FNSPID\\noticias_procesadas_FNSPID.csv"
]

# Función para parsear fechas y convertir a ET
def parse_to_et(date_str):
    try:
        dt = parser.parse(date_str)
    except Exception:
        return pd.NaT
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(ZoneInfo("America/New_York"))

dfs = []
for file in csv_files:
    df = pd.read_csv(file, encoding='utf-8')

    # Detectar la columna de fecha ('Fecha' o 'Date')
    for col in ['Fecha', 'Date']:
        if col in df.columns:
            df[col] = df[col].apply(parse_to_et)
            df = df.rename(columns={col: 'Fecha'})
            break
    else:
        raise ValueError(f"No se encontró columna de fecha en {file}")

    df = df.dropna(subset=['Fecha'])  # Eliminar fechas inválidas
    dfs.append(df)

# Concatenar y ordenar por fecha
df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.sort_values(by='Fecha').reset_index(drop=True)

# Guardar el CSV final
df_all.to_csv("C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\modelos_sentimiento\\noticias_procesadas_2020-2024.csv", index=False, encoding='utf-8')

print("CSV unificado con fechas en ET guardado correctamente.")

