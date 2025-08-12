import pandas as pd
from datetime import time

#Ruta del dataset con datos de pre y post mercado además de mercado intradía
input_file = f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_historico_5m\\QQQ_5min_all_data.csv"

# Ruta del archivo de salida filtrado con datos de mercado intradía solamente
output_file = f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_historico_5m\\QQQ_5min_intra_market_data.csv"

#Lectura del archivo y definición del horario de mercado (09:30 a 15:55 hora de Nueva York)
df = pd.read_csv(input_file, parse_dates=['timestamp'])
start_time = time(9, 30)
end_time = time(15, 55)

#Filtrado
df_filtrado = df[df['timestamp'].dt.time.between(start_time, end_time)]

#Guardado
df_filtrado.to_csv(output_file, index=False)
print(f"Archivo filtrado guardado en: {output_file}")
