# ================================================================
# Fase 2 (PA-41Tapyta): PCA + K-means con K=6
# Visualizaciones estáticas con matplotlib + seaborn
# Autor: Alcides Rojas (ajustado para benchmark PA-41Tapyta)
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
ruta_features = ruta_salida / "features_PA-41Tapyta.parquet"  # benchmark específico
ruta_salida.mkdir(parents=True, exist_ok=True)

# Parámetros
USAR_SEMILLA = True
SEMILLA = 123
UMBRAL_SD1 = 12.0            # alineado con distribución observada (pico ~15), más inclusivo que 15
FRACCION_MUESTRA = 0.25      # fracción para acelerar manteniendo representatividad
N_COMPONENTES_PCA_CLUST = 8  # más componentes para captar matices del sitio
K = 6                        # único valor de k a probar

# Features: compactos y complementarios (evita redundancia excesiva)
FEATURES_BASE = [f"mfcc_mean_{i}" for i in range(1, 6)]  # 1..5
FEATURES_SD = ["mfcc_sd_1", "mfcc_sd_2", "mfcc_sd_3"]

# Estética
sns.set_theme(context="talk", style="whitegrid", palette="tab20")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["savefig.dpi"] = 200

# -----------------------------
# 1) Carga de datos
# -----------------------------

if not ruta_features.exists():
    raise FileNotFoundError(f"No se encontró el parquet de Fase 1 (benchmark PA): {ruta_features}")

df = pd.read_parquet(ruta_features)

# -----------------------------
# 2) Selección de features y filtros acústicos
# -----------------------------

features = [f for f in FEATURES_BASE + FEATURES_SD if f in df.columns]
faltantes_base = [f for f in FEATURES_BASE if f not in df.columns]
if faltantes_base:
    raise ValueError(f"Faltan columnas base en el parquet: {faltantes_base}")

df_filtrado = df.copy()

# Filtro por variabilidad (excluir segmentos demasiado planos)
if "mfcc_sd_1" in df_filtrado.columns:
    df_filtrado = df_filtrado[df_filtrado["mfcc_sd_1"] >= UMBRAL_SD1]

# Muestreo para acelerar
df_filtrado = df_filtrado.sample(frac=FRACCION_MUESTRA, random_state=SEMILLA)

if len(df_filtrado) < 50:
    print("Advertencia: pocos segmentos tras los filtros.")

# -----------------------------
# 3) Escalado robusto y PCA
# -----------------------------

X = df_filtrado[features].dropna()
idx = X.index

scaler = RobustScaler()
X_std = scaler.fit_transform(X)

pca_full = PCA(n_components=N_COMPONENTES_PCA_CLUST, random_state=SEMILLA)
X_pca_full = pca_full.fit_transform(X_std)

# Para visualización en 2D (PC1-PC2)
X_pca_2d = X_pca_full[:, :2]
emb_df = pd.DataFrame({"DIM1": X_pca_2d[:, 0], "DIM2": X_pca_2d[:, 1]}, index=idx)
var_exp_2d = float(np.sum(pca_full.explained_variance_ratio_[:2]))
var_exp_total = float(np.sum(pca_full.explained_variance_ratio_))

# Sincronizar con df_filtrado
df_base_plot = df_filtrado.loc[idx].copy()
df_base_plot["DIM1"] = emb_df["DIM1"].values
df_base_plot["DIM2"] = emb_df["DIM2"].values

# -----------------------------
# 4) K-means en PCs altos y visualización
# -----------------------------

kmeans = KMeans(n_clusters=K, n_init=50, random_state=SEMILLA)
labels = kmeans.fit_predict(X_pca_full)

# Silhouette para evaluación
try:
    sil = silhouette_score(X_pca_full, labels)
except Exception:
    sil = np.nan

df_plot = df_base_plot.copy()
df_plot["cluster"] = labels

titulo = (
    f"PCA({N_COMPONENTES_PCA_CLUST}) + K-means (k={K}) | "
    f"sd1≥{UMBRAL_SD1} | semilla {SEMILLA} | "
    f"VarExp2D={var_exp_2d:.2f} | VarExpTotal={var_exp_total:.2f}"
)

# Scatter PC1-PC2 coloreado por cluster
fig, ax = plt.subplots()
sns.scatterplot(
    data=df_plot,
    x="DIM1", y="DIM2",
    hue="cluster",
    palette="tab20",
    s=18,
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
salida_png_scatter = ruta_salida / f"PA41_fase2_clusters_scatter_pca{N_COMPONENTES_PCA_CLUST}_k{K}.png"
plt.savefig(salida_png_scatter)
plt.close(fig)

# Boxplot por cluster (evita FutureWarning usando hue y legend=False)
fig2, ax2 = plt.subplots()
sns.boxplot(
    data=df_plot,
    x="cluster",
    y="mfcc_mean_1",
    hue="cluster",           # igualamos hue a x para colorear por cluster
    palette="tab20",
    dodge=False,
    ax=ax2
)
ax2.legend_.remove()          # equivalente a legend=False para evitar leyenda redundante
ax2.set_title(f"Distribución de mfcc_mean_1 por cluster (k={K})", fontsize=15)
ax2.set_xlabel("Cluster")
ax2.set_ylabel("mfcc_mean_1")
sns.despine()
plt.tight_layout()
salida_png_box_cluster = ruta_salida / f"PA41_fase2_boxplot_mfcc_mean1_por_cluster_k{K}.png"
plt.savefig(salida_png_box_cluster)
plt.close(fig2)

# Resumen por cluster (CSV)
cols_resumen = {
    "conteo": ("cluster", "size"),
    "mean1": ("mfcc_mean_1", "mean"),
    "mean2": ("mfcc_mean_2", "mean"),
    "mean3": ("mfcc_mean_3", "mean"),
}
if "mfcc_sd_1" in df_plot.columns:
    cols_resumen["sd1_prom"] = ("mfcc_sd_1", "mean")

resumen_cluster = df_plot.groupby("cluster").agg(**cols_resumen)
resumen_cluster.to_csv(ruta_salida / f"PA41_fase2_resumen_por_cluster_k{K}.csv")

# Exportar embedding + clusters para trazabilidad (CSV)
cols_export = ["DIM1", "DIM2", "cluster"] + [c for c in FEATURES_BASE + FEATURES_SD if c in df_plot.columns]
cols_export += [c for c in ["sitio", "archivo_origen", "tiempo_inicio", "tiempo_fin"] if c in df_plot.columns]
df_plot[cols_export].to_csv(ruta_salida / f"PA41_fase2_embedding_clusters_pca{N_COMPONENTES_PCA_CLUST}_k{K}.csv", index=False)

# -----------------------------
# 5) Reporte silhouette y varianza
# -----------------------------

df_sil = pd.DataFrame([{"k": K, "silhouette": sil, "var_exp_total": var_exp_total, "var_exp_2d": var_exp_2d}])
df_sil.to_csv(ruta_salida / "PA41_fase2_silhouette_scores_pca.csv", index=False)

print("Fase 2 (PA-41Tapyta) completada.")
print(f"- Scatter: {salida_png_scatter.name}")
print(f"- Boxplot: {salida_png_box_cluster.name}")
print(f"- Resumen clusters CSV: PA41_fase2_resumen_por_cluster_k{K}.csv")
print(f"- Embedding CSV: PA41_fase2_embedding_clusters_pca{N_COMPONENTES_PCA_CLUST}_k{K}.csv")
print(f"- Silhouette: PA41_fase2_silhouette_scores_pca.csv (sil={sil:.3f}, VarTotal={var_exp_total:.2f})")