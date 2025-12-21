# ================================================================
# Exploración visual de MFCCs (Fase 1)
# Autor: Alcides Rojas
# Objetivo: Explorar features.parquet con visualizaciones claras
# ================================================================

# --- Importaciones principales ---
import os
import pandas as pd
import numpy as np

# Librerías de visualización
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuración global de visualización ---
# Establecemos un estilo visual agradable y coherente para todos los gráficos
sns.set_theme(context="notebook", style="whitegrid", palette="viridis")
plt.rcParams["figure.figsize"] = (10, 6)   # tamaño por defecto de las figuras
plt.rcParams["figure.dpi"] = 120           # resolución por defecto
plt.rcParams["axes.titleweight"] = "bold"  # títulos en negrita
plt.rcParams["axes.labelsize"] = 11        # tamaño de etiquetas

# --- Parámetros del usuario ---
# Ruta al archivo features.parquet (ajustá si es necesario)
RUTA_PARQUET = r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local\resultados\features.parquet"

# Cantidad máxima de filas para muestrear en gráficos (evitar saturación)
# Si tenés millones de filas, graficar todas puede ser lento; usamos una muestra aleatoria.
MUESTRA_MAX = 200_000

# Sitios a mostrar en comparaciones (si None, usa todos)
SITIOS_SELECCIONADOS = None  # e.g., ["BO", "PA-41Tapyta"]

# --- Funciones auxiliares de visualización ---
def asegurar_columnas(df: pd.DataFrame, columnas: list[str]) -> list[str]:
    """
    Verifica que las columnas existan en el DataFrame y devuelve las disponibles.
    Sirve para evitar errores si alguna columna no está presente.
    """
    return [c for c in columnas if c in df.columns]

def muestrear_df(df: pd.DataFrame, n_max: int, sitios: list[str] | None = None) -> pd.DataFrame:
    """
    Devuelve una muestra del DataFrame para visualizaciones.
    - Si 'sitios' se especifica, filtra primero por esos sitios.
    - Luego toma una muestra aleatoria de hasta 'n_max' filas para graficar sin lentitud.
    """
    if sitios:
        df = df[df["sitio"].isin(sitios)]
    if len(df) > n_max:
        return df.sample(n_max, random_state=42)
    return df

def guardar_fig(nombre: str):
    """
    Guarda la figura actual en PNG con nombre especificado dentro de la carpeta 'figuras'.
    Útil para documentar resultados visuales.
    """
    os.makedirs("figuras", exist_ok=True)
    plt.savefig(os.path.join("figuras", f"{nombre}.png"), bbox_inches="tight")

# --- 1) Cargar el archivo 'features.parquet' ---
# Cargamos el dataset de segmentos con MFCCs. Requiere pyarrow o fastparquet instalado.
print("Cargando datos desde:", RUTA_PARQUET)
df = pd.read_parquet(RUTA_PARQUET)

# --- 2) Mostrar cuántos segmentos hay y qué columnas contiene ---
num_segmentos, num_columnas = df.shape
print(f"Segmentos (filas): {num_segmentos} | Columnas: {num_columnas}")
print("Columnas disponibles:")
print(df.columns.tolist())

# --- 3) Ver los primeros 5 registros ---
print("\nPrimeras 5 filas:")
print(df.head())

# --- 4) Graficar la distribución de 'mfcc_mean_1' ---
# Idea: un histograma con KDE nos muestra la forma de la distribución: simétrica, sesgada, outliers, etc.
cols_mfcc_mean = asegurar_columnas(df, [f"mfcc_mean_{i}" for i in range(1, 21)])
cols_mfcc_sd   = asegurar_columnas(df, [f"mfcc_sd_{i}" for i in range(1, 21)])

if "mfcc_mean_1" in cols_mfcc_mean:
    df_plot = muestrear_df(df, MUESTRA_MAX, SITIOS_SELECCIONADOS)
    plt.figure()
    sns.histplot(df_plot["mfcc_mean_1"], bins=60, kde=True, color="teal")
    plt.title("Distribución de mfcc_mean_1 (muestra)")
    plt.xlabel("Valor de mfcc_mean_1")
    plt.ylabel("Frecuencia")
    guardar_fig("hist_mfcc_mean_1")
    plt.show()
else:
    print("mfcc_mean_1 no está presente; se omite el histograma.")

# --- 5) Comparar por sitio (boxplot) para 'mfcc_mean_1' ---
# Un boxplot por 'sitio' permite ver diferencias acústicas entre lugares:
# medianas, dispersiones, outliers. Es útil para evaluar si un sitio suena "distinto".
if "mfcc_mean_1" in cols_mfcc_mean and "sitio" in df.columns:
    df_plot = muestrear_df(df, MUESTRA_MAX, SITIOS_SELECCIONADOS)
    # Para evitar sitios con muy pocas observaciones, filtramos los top sitios por cantidad
    conteos_por_sitio = df_plot["sitio"].value_counts().head(20).index.tolist()
    df_plot = df_plot[df_plot["sitio"].isin(conteos_por_sitio)]

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_plot, x="sitio", y="mfcc_mean_1", showfliers=False)
    sns.stripplot(data=df_plot.sample(min(len(df_plot), 5000), random_state=42),
                  x="sitio", y="mfcc_mean_1", color="black", alpha=0.15, jitter=True)
    plt.title("mfcc_mean_1 por sitio (boxplot + puntos de muestra)")
    plt.xlabel("Sitio")
    plt.ylabel("mfcc_mean_1")
    plt.xticks(rotation=45, ha="right")
    guardar_fig("boxplot_mfcc_mean_1_por_sitio")
    plt.show()
else:
    print("No se puede generar boxplot: faltan columnas 'sitio' o 'mfcc_mean_1'.")

# --- 6) Correlación entre mfcc_mean_1 a mfcc_mean_5 (heatmap) ---
# El heatmap de correlación muestra si los coeficientes están muy relacionados (redundancia)
cols_corr_5 = asegurar_columnas(df, [f"mfcc_mean_{i}" for i in range(1, 6)])
if len(cols_corr_5) >= 2:
    df_plot = muestrear_df(df[cols_corr_5], MUESTRA_MAX)
    corr = df_plot.corr()
    plt.figure()
    sns.heatmap(corr, annot=True, cmap="mako", vmin=-1, vmax=1)
    plt.title("Correlación entre mfcc_mean_1 a mfcc_mean_5")
    guardar_fig("heatmap_corr_mfcc_mean_1_5")
    plt.show()
else:
    print("No hay suficientes columnas mfcc_mean_1..5 para correlación.")

# --- 7) Detectar valores extremos o silencios con histogramas ('mfcc_mean_1' y 'mfcc_sd_1') ---
# Idea: 'mfcc_sd_1' muy bajo puede indicar segmentos "planos" (posibles silencios o señales constantes).
if "mfcc_mean_1" in cols_mfcc_mean and "mfcc_sd_1" in cols_mfcc_sd:
    df_plot = muestrear_df(df, MUESTRA_MAX, SITIOS_SELECCIONADOS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df_plot["mfcc_mean_1"], bins=60, kde=True, ax=axes[0], color="#5CC8FF")
    axes[0].set_title("Distribución de mfcc_mean_1")
    axes[0].set_xlabel("Valor")
    axes[0].set_ylabel("Frecuencia")

    sns.histplot(df_plot["mfcc_sd_1"], bins=60, kde=True, ax=axes[1], color="#FF8C69")
    axes[1].set_title("Distribución de mfcc_sd_1 (variabilidad)")
    axes[1].set_xlabel("Valor")
    axes[1].set_ylabel("Frecuencia")

    plt.suptitle("Detección de extremos: medias y desviaciones (muestra)", y=1.02, fontsize=12, fontweight="bold")
    guardar_fig("hist_mfcc_mean_sd_1")
    plt.tight_layout()
    plt.show()
else:
    print("No se pueden graficar histogramas combinados: faltan 'mfcc_mean_1' o 'mfcc_sd_1'.")

# --- 8) Diferencias acústicas entre sitios (boxplots de 'mfcc_mean_1' y 'mfcc_mean_2') ---
# Comparamos los primeros coeficientes por sitio. Los coeficientes bajos (1, 2) capturan el contorno espectral general.
cols_mfcc_12 = asegurar_columnas(df, ["mfcc_mean_1", "mfcc_mean_2"])
if "sitio" in df.columns and len(cols_mfcc_12) == 2:
    df_plot = muestrear_df(df, MUESTRA_MAX, SITIOS_SELECCIONADOS)
    sitios_top = df_plot["sitio"].value_counts().head(15).index.tolist()
    df_plot = df_plot[df_plot["sitio"].isin(sitios_top)]

    # Boxplot lado a lado: dos subgráficos para comparar coeficientes 1 y 2
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(data=df_plot, x="sitio", y="mfcc_mean_1", ax=axes[0], showfliers=False)
    axes[0].set_title("mfcc_mean_1 por sitio")
    axes[0].set_xlabel("Sitio"); axes[0].set_ylabel("mfcc_mean_1")
    axes[0].tick_params(axis="x", rotation=45)

    sns.boxplot(data=df_plot, x="sitio", y="mfcc_mean_2", ax=axes[1], showfliers=False)
    axes[1].set_title("mfcc_mean_2 por sitio")
    axes[1].set_xlabel("Sitio"); axes[1].set_ylabel("mfcc_mean_2")
    axes[1].tick_params(axis="x", rotation=45)

    plt.suptitle("Diferencias acústicas por sitio (coeficientes MFCC bajos)", y=1.02, fontsize=12, fontweight="bold")
    guardar_fig("boxplot_mfcc_mean_1_2_por_sitio")
    plt.tight_layout()
    plt.show()
else:
    print("No se puede comparar sitios: faltan 'sitio' o 'mfcc_mean_1/2'.")

# --- 9) Redundancia o complementariedad (heatmap mfcc_mean_1..10) ---
# Un heatmap con 10 coeficientes permite ver bloques correlacionados (redundancia) o independientes (complementariedad).
cols_corr_10 = asegurar_columnas(df, [f"mfcc_mean_{i}" for i in range(1, 11)])
if len(cols_corr_10) >= 3:
    df_plot = muestrear_df(df[cols_corr_10], MUESTRA_MAX)
    corr = df_plot.corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=False, cmap="Spectral", center=0, vmin=-1, vmax=1)
    plt.title("Correlación entre mfcc_mean_1..10")
    guardar_fig("heatmap_corr_mfcc_mean_1_10")
    plt.show()
else:
    print("No hay suficientes columnas mfcc_mean_1..10 para correlación.")

# --- 10) Sitios con sonidos más agudos o graves (heurística con 'mfcc_mean_1') ---
# Nota: En MFCCs, los primeros coeficientes reflejan el contorno espectral global.
# Comparar promedios por sitio de 'mfcc_mean_1' es una heurística para ver diferencias tonales generales.
# No implica directamente "grave vs agudo", pero orienta sobre el perfil espectral.

if "mfcc_mean_1" in df.columns and "sitio" in df.columns:
    # Calculamos el promedio de mfcc_mean_1 por sitio
    promedio_por_sitio = (
        df.groupby("sitio", as_index=False)["mfcc_mean_1"]
        .mean()
        .rename(columns={"mfcc_mean_1": "promedio_mfcc_mean_1"})
    )

    # Ordenamos sitios del valor más bajo al más alto
    promedio_por_sitio = promedio_por_sitio.sort_values("promedio_mfcc_mean_1")

    # Visualización: barras ordenadas para lectura clara
    plt.figure(figsize=(12, 6))
    sns.barplot(data=promedio_por_sitio, x="sitio", y="promedio_mfcc_mean_1", palette="crest")
    plt.title("Promedio de mfcc_mean_1 por sitio (ordenado)")
    plt.xlabel("Sitio"); plt.ylabel("Promedio de mfcc_mean_1")
    plt.xticks(rotation=45, ha="right")
    guardar_fig("barras_promedio_mfcc_mean_1_por_sitio")
    plt.show()

    # Impresión de los 10 sitios con valores más bajos y más altos
    print("\nTop 10 sitios con promedio más bajo de mfcc_mean_1 (perfil espectral distinto):")
    print(promedio_por_sitio.head(10))

    print("\nTop 10 sitios con promedio más alto de mfcc_mean_1:")
    print(promedio_por_sitio.tail(10))
else:
    print("No se puede calcular promedios por sitio: faltan 'sitio' o 'mfcc_mean_1'.")

# --- Consejos interpretativos (impresos, no gráficos) ---
print("\nInterpretación orientativa:")
print("- Los histogramas te muestran la forma general de los coeficientes (simetría, sesgo, outliers).")
print("- Los boxplots por sitio comparan la distribución: medianas distintas sugieren paisajes sonoros diferentes.")
print("- Los heatmaps de correlación revelan redundancia: coeficientes muy correlacionados pueden reducirse.")
print("- La barra de promedios por sitio es una heurística: diferencia tonal general, no clasifica 'grave/agudo' de manera directa.")
print("- Si ves mfcc_sd_1 muy bajo en muchos segmentos, puede indicar tramos silenciosos o con poca variación espectral.")
