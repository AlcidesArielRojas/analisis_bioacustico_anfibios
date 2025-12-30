# ================================================================
# Exploración visual de MFCCs (Fase 1 Benchmark PA-41Tapyta)
# Autor: Alcides Rojas
# Objetivo: Explorar features_PA-41Tapyta.parquet con visualizaciones claras
# ================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuración global de visualización ---
sns.set_theme(context="notebook", style="whitegrid", palette="viridis")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11

# --- Parámetros del usuario ---
# Ruta al archivo parquet del benchmark
RUTA_PARQUET = r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local\resultados\features_PA-41Tapyta.parquet"

# Cantidad máxima de filas para muestrear en gráficos
MUESTRA_MAX = 200_000

# Sitios a mostrar (en este caso solo PA-41Tapyta)
SITIOS_SELECCIONADOS = ["PA-41Tapyta"]

# --- Funciones auxiliares ---
def asegurar_columnas(df: pd.DataFrame, columnas: list[str]) -> list[str]:
    return [c for c in columnas if c in df.columns]

def muestrear_df(df: pd.DataFrame, n_max: int, sitios: list[str] | None = None) -> pd.DataFrame:
    if sitios:
        df = df[df["sitio"].isin(sitios)]
    if len(df) > n_max:
        return df.sample(n_max, random_state=42)
    return df

def guardar_fig(nombre: str):
    os.makedirs("figuras_PA-41Tapyta", exist_ok=True)
    plt.savefig(os.path.join("figuras_PA-41Tapyta", f"{nombre}.png"), bbox_inches="tight")

# --- 1) Cargar datos ---
print("Cargando datos desde:", RUTA_PARQUET)
df = pd.read_parquet(RUTA_PARQUET)

# --- 2) Info básica ---
num_segmentos, num_columnas = df.shape
print(f"Segmentos: {num_segmentos} | Columnas: {num_columnas}")
print("Columnas disponibles:", df.columns.tolist())

# --- 3) Primeras filas ---
print("\nPrimeras 5 filas:")
print(df.head())

# --- 4) Histograma de mfcc_mean_1 ---
cols_mfcc_mean = asegurar_columnas(df, [f"mfcc_mean_{i}" for i in range(1, 21)])
cols_mfcc_sd   = asegurar_columnas(df, [f"mfcc_sd_{i}" for i in range(1, 21)])

if "mfcc_mean_1" in cols_mfcc_mean:
    df_plot = muestrear_df(df, MUESTRA_MAX, SITIOS_SELECCIONADOS)
    plt.figure()
    sns.histplot(df_plot["mfcc_mean_1"], bins=60, kde=True, color="teal")
    plt.title("Distribución de mfcc_mean_1 (PA-41Tapyta)")
    plt.xlabel("Valor de mfcc_mean_1")
    plt.ylabel("Frecuencia")
    guardar_fig("hist_mfcc_mean_1")
    plt.show()

# --- 5) Boxplot por sitio ---
if "mfcc_mean_1" in cols_mfcc_mean and "sitio" in df.columns:
    df_plot = muestrear_df(df, MUESTRA_MAX, SITIOS_SELECCIONADOS)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_plot, x="sitio", y="mfcc_mean_1", showfliers=False)
    plt.title("mfcc_mean_1 por sitio (PA-41Tapyta)")
    plt.xlabel("Sitio")
    plt.ylabel("mfcc_mean_1")
    plt.xticks(rotation=45, ha="right")
    guardar_fig("boxplot_mfcc_mean_1_por_sitio")
    plt.show()

# --- 6) Correlación mfcc_mean_1..5 ---
cols_corr_5 = asegurar_columnas(df, [f"mfcc_mean_{i}" for i in range(1, 6)])
if len(cols_corr_5) >= 2:
    df_plot = muestrear_df(df[cols_corr_5], MUESTRA_MAX)
    corr = df_plot.corr()
    plt.figure()
    sns.heatmap(corr, annot=True, cmap="mako", vmin=-1, vmax=1)
    plt.title("Correlación entre mfcc_mean_1..5 (PA-41Tapyta)")
    guardar_fig("heatmap_corr_mfcc_mean_1_5")
    plt.show()

# --- 7) Histogramas de mfcc_mean_1 y mfcc_sd_1 ---
if "mfcc_mean_1" in cols_mfcc_mean and "mfcc_sd_1" in cols_mfcc_sd:
    df_plot = muestrear_df(df, MUESTRA_MAX, SITIOS_SELECCIONADOS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df_plot["mfcc_mean_1"], bins=60, kde=True, ax=axes[0], color="#5CC8FF")
    axes[0].set_title("Distribución de mfcc_mean_1")
    sns.histplot(df_plot["mfcc_sd_1"], bins=60, kde=True, ax=axes[1], color="#FF8C69")
    axes[1].set_title("Distribución de mfcc_sd_1")
    plt.suptitle("Distribución de medias y desviaciones (PA-41Tapyta)", y=1.02, fontsize=12, fontweight="bold")
    guardar_fig("hist_mfcc_mean_sd_1")
    plt.tight_layout()
    plt.show()

# --- 8) Heatmap mfcc_mean_1..10 ---
cols_corr_10 = asegurar_columnas(df, [f"mfcc_mean_{i}" for i in range(1, 11)])
if len(cols_corr_10) >= 3:
    df_plot = muestrear_df(df[cols_corr_10], MUESTRA_MAX)
    corr = df_plot.corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=False, cmap="Spectral", center=0, vmin=-1, vmax=1)
    plt.title("Correlación entre mfcc_mean_1..10 (PA-41Tapyta)")
    guardar_fig("heatmap_corr_mfcc_mean_1_10")
    plt.show()

# --- 9) Promedio de mfcc_mean_1 por sitio ---
if "mfcc_mean_1" in df.columns and "sitio" in df.columns:
    promedio_por_sitio = (
        df.groupby("sitio", as_index=False)["mfcc_mean_1"]
        .mean()
        .rename(columns={"mfcc_mean_1": "promedio_mfcc_mean_1"})
    )
    promedio_por_sitio = promedio_por_sitio.sort_values("promedio_mfcc_mean_1")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=promedio_por_sitio, x="sitio", y="promedio_mfcc_mean_1", palette="crest")
    plt.title("Promedio de mfcc_mean_1 por sitio (PA-41Tapyta)")
    plt.xlabel("Sitio"); plt.ylabel("Promedio de mfcc_mean_1")
    plt.xticks(rotation=45, ha="right")
    guardar_fig("barras_promedio_mfcc_mean_1_por_sitio")
    plt.show()

    print("\nPromedio de mfcc_mean_1 por sitio:")
    print(promedio_por_sitio)

print("\nInterpretación orientativa:")
print("- Histogramas muestran forma general de coeficientes.")
print("- Boxplots comparan distribución por sitio.")
print("- Heatmaps revelan redundancia o complementariedad.")
print("- Promedios por sitio sugieren diferencias tonales generales.")
