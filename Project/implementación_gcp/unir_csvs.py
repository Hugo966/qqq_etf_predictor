import pandas as pd

#Datasets anuales a unir
csv_files = [
    f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_historico_5m\\QQQ_5min_data_2020.csv",
    f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_historico_5m\\QQQ_5min_data_2021.csv",
    f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_historico_5m\\QQQ_5min_data_2022.csv",
    f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_historico_5m\\QQQ_5min_data_2023.csv",
    f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_historico_5m\\QQQ_5min_data_2024.csv",
    f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_historico_5m\\QQQ_5min_data_2025.csv"
]

#Nombre del archivo combinado
output_file = f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_historico_5m\\QQQ_5min_all_data.csv"

#Lectura y unión de los CSVs
dataframes = []
for i, file in enumerate(csv_files):
    df = pd.read_csv(file)
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)

#Guardado del resultado
combined_df.to_csv(output_file, index=False)
print(f"Archivo combinado creado: {output_file}")
