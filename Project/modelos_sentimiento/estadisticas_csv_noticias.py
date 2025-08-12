import pandas as pd
import matplotlib.pyplot as plt
from zoneinfo import ZoneInfo

# Ruta del CSV unificado
csv_path = "C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\modelos_sentimiento\\noticias_procesadas_2020-2024.csv"

# 1) Cargar el CSV
df = pd.read_csv(csv_path, encoding='utf-8')

# 2) Convertir 'Fecha' a datetime con utc=True para evitar el warning y parsear bien zonas horarias
df['Fecha'] = pd.to_datetime(df['Fecha'], utc=True, errors='coerce')

# 3) Convertir de UTC a ET (hora de Nueva York)
df['Fecha'] = df['Fecha'].dt.tz_convert(ZoneInfo("America/New_York"))

# 4) Eliminar filas sin fecha válida
df = df.dropna(subset=['Fecha'])

# 5) Extraer año y mes
df['Año'] = df['Fecha'].dt.year
df['Mes'] = df['Fecha'].dt.to_period('M')

# 6) Estadísticas generales
total_notas = len(df)
counts_per_year = df['Año'].value_counts().sort_index()
counts_per_month = df['Mes'].value_counts().sort_index()

# 7) Mostrar por consola
print(f"Total de noticias: {total_notas}\n")
print("Noticias por año:")
print(counts_per_year.to_string(), "\n")
print("Descriptivo de Sentiment_score:")
print(df['Sentiment_score'].describe(), "\n")

# --- Agregación quincenal (cada 2 semanas) para Sentiment Score ---

# Obtener inicio de semana (lunes) para cada fecha
df['Fecha_bi_semana'] = df['Fecha'].dt.to_period('W').apply(lambda r: r.start_time)

# Crear grupo bimensual para agrupar cada 2 semanas (14 días)
df['Bi_semana_group'] = (df['Fecha_bi_semana'].map(pd.Timestamp.toordinal) // 14)

# Agrupar por el grupo quincenal y calcular media de Sentiment_score y fecha mínima del grupo
sentiment_bi_semanal = df.groupby('Bi_semana_group').agg({
    'Fecha_bi_semana': 'min',
    'Sentiment_score': 'mean'
}).reset_index(drop=True)

# Crear rango completo quincenal para evitar huecos en la serie temporal
fechas_bi_completas = pd.date_range(
    start=sentiment_bi_semanal['Fecha_bi_semana'].min(),
    end=sentiment_bi_semanal['Fecha_bi_semana'].max() + pd.Timedelta(days=7),
    freq='14D'
)

# Reindexar para incluir todas las quincenas (períodos de 2 semanas)
sentiment_bi_semanal = sentiment_bi_semanal.set_index('Fecha_bi_semana').reindex(fechas_bi_completas)

# Rellenar valores faltantes con forward fill para no tener huecos en la gráfica
sentiment_bi_semanal['Sentiment_score'] = sentiment_bi_semanal['Sentiment_score'].fillna(method='ffill')

# Resetear índice para graficar
sentiment_bi_semanal = sentiment_bi_semanal.reset_index()
sentiment_bi_semanal.rename(columns={'index': 'Fecha_bi_semana'}, inplace=True)

# --- Gráfica evolución quincenal del Sentiment Score ---
plt.figure(figsize=(12, 5))
plt.plot(sentiment_bi_semanal['Fecha_bi_semana'], sentiment_bi_semanal['Sentiment_score'],
         marker='o', linestyle='-', markersize=5, alpha=0.85, color='tab:purple')
plt.title('Evolución quincenal (cada 2 semanas) del Sentiment Score medio')
plt.xlabel('Fecha inicio quincena')
plt.ylabel('Sentiment Score medio quincenal')
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

# 8) Histograma de distribución de Sentiment_score
plt.figure(figsize=(8, 4))
plt.hist(df['Sentiment_score'].dropna(), bins=20, color='skyblue', edgecolor='black')
plt.title('Distribución de Sentiment_score')
plt.xlabel('Sentiment_score')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# 9) Gráfico de barras de noticias por año
plt.figure(figsize=(6, 4))
counts_per_year.plot(kind='bar', color='orange', edgecolor='black')
plt.title('Número de noticias por año')
plt.xlabel('Año')
plt.ylabel('Número de noticias')
plt.tight_layout()
plt.show()

# 10) Gráfico de línea mensual (matplotlib)
monthly_df = counts_per_month.reset_index()
monthly_df.columns = ['Mes', 'Cantidad']
monthly_df['Mes'] = monthly_df['Mes'].dt.to_timestamp()

plt.figure(figsize=(12, 5))
plt.plot(monthly_df['Mes'], monthly_df['Cantidad'], marker='o', linestyle='-', color='green')
plt.title('Evolución mensual de noticias (2020-2024)')
plt.xlabel('Mes')
plt.ylabel('Número de noticias')
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()
