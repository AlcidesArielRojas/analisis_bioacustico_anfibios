# ================================================================
# Fase 2 (no interactiva): PCA (PCs altos) + K-means
# Visualizaciones estáticas con matplotlib + seaborn
# Autor: Alcides Rojas (adaptado)
# ================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------------
# Configuración general
# -----------------------------

ruta_salida = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local\resultados")
ruta_features = ruta_salida / "features.parquet"
ruta_salida.mkdir(parents=True, exist_ok=True)

# Mantenemos coherencia con tus parámetros base
METODO_DIM = "PCA"             # Usamos PCA
USAR_SEMILLA = True            # Reproducible
SEMILLA = 123
EXCLUIR_BO = False
PRIORIZAR_SOLO_PA = False
UMBRAL_SD1 = 10.0              # Restaurado
FRACCION_MUESTRA = 0.3         # Usar una fracción para pruebas
N_COMPONENTES_PCA_CLUST = 6   # PCs para clustering (alto)
KS_A_PROBAR = [4, 6, 8, 10]    # Valores de k para comparar

FEATURES_BASE = ["mfcc_mean_1", "mfcc_mean_2", "mfcc_mean_3"]
INCLUIR_SD = True
FEATURES_SD = ["mfcc_sd_1", "mfcc_sd_2", "mfcc_sd_3"] if INCLUIR_SD else []

# Estética
sns.set_theme(context="talk", style="whitegrid", palette="tab20")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["savefig.dpi"] = 200

# -----------------------------
# 1) Carga de datos
# -----------------------------

if not ruta_features.exists():
    raise FileNotFoundError(f"No se encontró el parquet de Fase 1: {ruta_features}")

df = pd.read_parquet(ruta_features)

# -----------------------------
# 2) Selección de features y filtros acústicos
# -----------------------------

features = [f for f in FEATURES_BASE + FEATURES_SD if f in df.columns]
if len([f for f in FEATURES_BASE if f in df.columns]) < len(FEATURES_BASE):
    raise ValueError("Faltan columnas mfcc_mean_1..3 en el parquet.")

df_filtrado = df.copy()

# Excluir/filtrar sitios si aplica
if EXCLUIR_BO and "sitio" in df_filtrado.columns:
    df_filtrado = df_filtrado[~df_filtrado["sitio"].str.contains("BO", case=False, na=False)]

if PRIORIZAR_SOLO_PA and "sitio" in df_filtrado.columns:
    df_filtrado = df_filtrado[df_filtrado["sitio"].str.contains("PA", case=False, na=False)]

# Filtro por variabilidad
if "mfcc_sd_1" in df_filtrado.columns:
    df_filtrado = df_filtrado[df_filtrado["mfcc_sd_1"] >= UMBRAL_SD1]

# Usar fracción para pruebas
df_filtrado = df_filtrado.sample(frac=FRACCION_MUESTRA, random_state=SEMILLA)

if len(df_filtrado) < 50:
    print("Advertencia: pocos segmentos tras los filtros.")

# -----------------------------
# 3) Escalado robusto y PCA
# -----------------------------

X = df_filtrado[features].dropna()
idx = X.index

# Escalado robusto (más resistente a outliers)
scaler = RobustScaler()
X_std = scaler.fit_transform(X)

# PCA con más componentes para clustering
pca_full = PCA(n_components=N_COMPONENTES_PCA_CLUST, random_state=SEMILLA)
X_pca_full = pca_full.fit_transform(X_std)

# Para visualización en 2D, usamos PC1-PC2
X_pca_2d = X_pca_full[:, :2]
emb_df = pd.DataFrame({"DIM1": X_pca_2d[:, 0], "DIM2": X_pca_2d[:, 1]}, index=idx)
var_exp_2d = float(np.sum(pca_full.explained_variance_ratio_[:2]))

# Sincronizar con df_filtrado
df_base_plot = df_filtrado.loc[idx].copy()
df_base_plot["DIM1"] = emb_df["DIM1"].values
df_base_plot["DIM2"] = emb_df["DIM2"].values

# -----------------------------
# 4) Bucle de K-means en PCs altos y visualización en 2D
# -----------------------------

resultados_silhouette = []

for k in KS_A_PROBAR:
    # K-means sobre el espacio PCA de mayor dimensión
    kmeans = KMeans(n_clusters=k, n_init=50, random_state=SEMILLA)
    labels = kmeans.fit_predict(X_pca_full)

    # Silhouette para evaluación objetiva
    try:
        sil = silhouette_score(X_pca_full, labels)
    except Exception:
        sil = np.nan  # por si el tamaño o la distribución impiden cálculo
    resultados_silhouette.append({"k": k, "silhouette": sil})

    # Construir df para plot/export por k
    df_plot_k = df_base_plot.copy()
    df_plot_k["cluster"] = labels

    # Scatter PC1-PC2 coloreado por cluster
    titulo = (
        f"PCA({N_COMPONENTES_PCA_CLUST}) + K-means (k={k}) | "
        f"sd1≥{UMBRAL_SD1} | "
        f"{'solo PA | ' if PRIORIZAR_SOLO_PA else ''}"
        f"{'sin BO | ' if EXCLUIR_BO else ''}"
        f"semilla {SEMILLA} | VarExp2D={var_exp_2d:.2f}"
    )

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df_plot_k,
        x="DIM1", y="DIM2",
        hue="cluster",
        palette="tab20",
        s=20,
        linewidth=0,
        alpha=0.85,
        ax=ax
    )
    ax.set_title(titulo, fontsize=16)
    ax.set_xlabel("Componente 1", fontsize=12)
    ax.set_ylabel("Componente 2", fontsize=12)
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    sns.despine()
    plt.tight_layout()

    # Guardar sin pisar: sufijo por k
    salida_png_scatter = ruta_salida / f"fase2_clusters_scatter_pca20_k{k}.png"
    plt.savefig(salida_png_scatter)
    plt.close(fig)

    # Boxplot por cluster para inspección
    fig2, ax2 = plt.subplots()
    sns.boxplot(
        data=df_plot_k,
        x="cluster",
        y="mfcc_mean_1",
        palette="tab20",
        ax=ax2
    )
    ax2.set_title(f"Distribución de mfcc_mean_1 por cluster (k={k})", fontsize=15)
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("mfcc_mean_1")
    sns.despine()
    plt.tight_layout()
    salida_png_box_cluster = ruta_salida / f"fase2_boxplot_mfcc_mean1_por_cluster_k{k}.png"
    plt.savefig(salida_png_box_cluster)
    plt.close(fig2)

    # Boxplot por sitio (si existe) para validar mezcla por cluster
    if "sitio" in df_plot_k.columns:
        fig3, ax3 = plt.subplots(figsize=(14, 8))
        sns.boxplot(
            data=df_plot_k,
            x="sitio",
            y="mfcc_mean_1",
            palette="tab20",
            ax=ax3
        )
        ax3.set_title(f"Distribución de mfcc_mean_1 por sitio (k={k})", fontsize=15)
        ax3.set_xlabel("Sitio")
        ax3.set_ylabel("mfcc_mean_1")
        plt.xticks(rotation=45, ha="right")
        sns.despine()
        plt.tight_layout()
        salida_png_box_sitio = ruta_salida / f"fase2_boxplot_mfcc_mean1_por_sitio_k{k}.png"
        plt.savefig(salida_png_box_sitio)
        plt.close(fig3)

    # Resumen por cluster (CSV) con sufijo k
    cols_resumen = {
        "conteo": ("cluster", "size"),
        "mean1": ("mfcc_mean_1", "mean"),
        "mean2": ("mfcc_mean_2", "mean"),
        "mean3": ("mfcc_mean_3", "mean"),
    }
    if "mfcc_sd_1" in df_plot_k.columns:
        cols_resumen["sd1_prom"] = ("mfcc_sd_1", "mean")

    resumen_cluster = df_plot_k.groupby("cluster").agg(**cols_resumen)
    resumen_cluster.to_csv(ruta_salida / f"fase2_resumen_por_cluster_k{k}.csv")

    # Resumen por sitio (CSV) con sufijo k
    if "sitio" in df_plot_k.columns:
        cols_sitio = {
            "conteo": ("sitio", "size"),
            "mediana_mean1": ("mfcc_mean_1", "median"),
            "mediana_mean2": ("mfcc_mean_2", "median"),
        }
        if "mfcc_sd_1" in df_plot_k.columns:
            cols_sitio["sd1_prom"] = ("mfcc_sd_1", "mean")

        resumen_sitio = df_plot_k.groupby("sitio").agg(**cols_sitio)
        resumen_sitio.to_csv(ruta_salida / f"fase2_resumen_por_sitio_k{k}.csv")

    # Exportar embedding + clusters para trazabilidad (CSV) con sufijo k
    cols_export = ["DIM1", "DIM2", "cluster"] + [c for c in FEATURES_BASE + FEATURES_SD if c in df_plot_k.columns]
    cols_export += [c for c in ["sitio", "archivo_origen", "tiempo_inicio", "tiempo_fin"] if c in df_plot_k.columns]
    df_plot_k[cols_export].to_csv(ruta_salida / f"fase2_embedding_clusters_pca20_k{k}.csv", index=False)

# -----------------------------
# 5) Reporte agregado de silhouette
# -----------------------------

df_sil = pd.DataFrame(resultados_silhouette)
df_sil.to_csv(ruta_salida / "fase2_silhouette_scores_pca20.csv", index=False)

print("Fase 2 no interactiva completada.")
print("Guardados: scatter, boxplots y CSVs con sufijo por k, más silhouette global.")