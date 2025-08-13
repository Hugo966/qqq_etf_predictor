import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#Ruta al archivo JSON con las noticias
ruta_json = "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\QQQ_sentimiento_Marketaux\\noticias_mes.json"
with open(ruta_json, "r", encoding="utf-8") as f:
    valores = json.load(f)

#Rango de fechas
fechas = pd.date_range(start="2020-01-01", periods=len(valores), freq="MS")

df = pd.DataFrame({"Fecha": fechas, "Valor": valores})
#Fecha a números para matplotlib
date_nums = mdates.date2num(df["Fecha"])

#Histograma
plt.figure(figsize=(12, 6))
plt.hist(
    date_nums,
    bins=date_nums,
    weights=df["Valor"],
    edgecolor='black',
    align='mid'
)

#Formato eje X
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)

plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.title("Histograma Temporal de Valores Mensuales (ene 2020 - dic 2025)")
plt.tight_layout()
plt.show()
