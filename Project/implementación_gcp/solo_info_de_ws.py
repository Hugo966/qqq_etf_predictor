import pandas as pd

#Carga el CSV original
df = pd.read_csv(f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_historico_5m\\QQQ_5min_intra_market_data.csv")

#Columnas a eliminar
cols_to_drop = ['high', 'low', 'volume', 'sentiment_score']

#Eliminación de columnas
df = df.drop(columns=cols_to_drop)

#Guardado el CSV limpio
df.to_csv(f"C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_historico_5m\\QQQ_5min_intra_market_only_ws_data.csv", index=False)