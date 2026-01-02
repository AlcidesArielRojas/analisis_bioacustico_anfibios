# ================================================================
# Fase 2 extendida: PA-41Tapyta
# PCA y UMAP (2D/3D), clustering con y sin reducción,
# silhouette global y por cluster, Plotly 3D interactivo con hover
# ================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# UMAP y Plotly (instalar si falta: pip install umap-learn plotly)
import umap
import plotly.express as px

# -----------------------------
# Configuración general
# -----------------------------

ruta_salida = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local\resultados")
ruta_features = ruta_salida / "features_PA-41Tapyta.parquet"
ruta_salida.mkdir(parents=True, exist_ok=True)

SEMILLA = 123
UMBRAL_SD1 = 12.0
FRACCION_MUESTRA = 0.25
N_COMPONENTES_PCA_CLUST = 8
K = 6

# UMAP params
UMAP_N_NEIGHBORS = 14
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 3  # para 3D interactivo y también derivar 2D

FEATURES_BASE = [f"mfcc_mean_{i}" for i in range(1, 5 + 1)]  # 1..5
FEATURES_SD = ["mfcc_sd_1", "mfcc_sd_2", "mfcc_sd_3"]

sns.set_theme(context="talk", style="whitegrid", palette="tab20")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["savefig.dpi"] = 200

# -----------------------------
# 1) Carga y filtrado
# -----------------------------

if not ruta_features.exists():
    raise FileNotFoundError(f"No se encontró el parquet: {ruta_features}")

df = pd.read_parquet(ruta_features)
features = [f for f in FEATURES_BASE + FEATURES_SD if f in df.columns]
faltantes_base = [f for f in FEATURES_BASE if f not in df.columns]
if faltantes_base:
    raise ValueError(f"Faltan columnas base: {faltantes_base}")

df_filtrado = df.copy()
if "mfcc_sd_1" in df_filtrado.columns:
    df_filtrado = df_filtrado[df_filtrado["mfcc_sd_1"] >= UMBRAL_SD1]

df_filtrado = df_filtrado.sample(frac=FRACCION_MUESTRA, random_state=SEMILLA)
X = df_filtrado[features].dropna()
idx = X.index

if len(X) < 50:
    print("Advertencia: pocos segmentos tras los filtros.")

# -----------------------------
# 2) Escalado
# -----------------------------

scaler = RobustScaler()
X_std = scaler.fit_transform(X)

# -----------------------------
# 3) PCA (2D/3D) y clustering
# -----------------------------

pca = PCA(n_components=N_COMPONENTES_PCA_CLUST, random_state=SEMILLA)
X_pca = pca.fit_transform(X_std)

# Para plots: 2D y 3D
X_pca_2d = X_pca[:, :2]
X_pca_3d = X_pca[:, :3]

var_exp_2d = float(np.sum(pca.explained_variance_ratio_[:2]))
var_exp_total = float(np.sum(pca.explained_variance_ratio_))

kmeans_pca = KMeans(n_clusters=K, n_init=50, random_state=SEMILLA)
labels_pca = kmeans_pca.fit_predict(X_pca)
sil_pca = silhouette_score(X_pca, labels_pca)
sil_samples_pca = silhouette_samples(X_pca, labels_pca)

df_pca = df_filtrado.loc[idx].copy()
df_pca["DIM1"] = X_pca_2d[:, 0]
df_pca["DIM2"] = X_pca_2d[:, 1]
df_pca["PC1"] = X_pca_3d[:, 0]
df_pca["PC2"] = X_pca_3d[:, 1]
df_pca["PC3"] = X_pca_3d[:, 2]
df_pca["cluster"] = labels_pca
df_pca["silhouette"] = sil_samples_pca

# -----------------------------
# 4) UMAP (2D/3D) y clustering
# -----------------------------

umap_model = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    n_components=UMAP_N_COMPONENTS,
    random_state=SEMILLA
)
X_umap = umap_model.fit_transform(X_std)

X_umap_2d = X_umap[:, :2]
X_umap_3d = X_umap[:, :3]

kmeans_umap = KMeans(n_clusters=K, n_init=50, random_state=SEMILLA)
labels_umap = kmeans_umap.fit_predict(X_umap)
sil_umap = silhouette_score(X_umap, labels_umap)
sil_samples_umap = silhouette_samples(X_umap, labels_umap)

df_umap = df_filtrado.loc[idx].copy()
df_umap["DIM1"] = X_umap_2d[:, 0]
df_umap["DIM2"] = X_umap_2d[:, 1]
df_umap["U1"] = X_umap_3d[:, 0]
df_umap["U2"] = X_umap_3d[:, 1]
df_umap["U3"] = X_umap_3d[:, 2]
df_umap["cluster"] = labels_umap
df_umap["silhouette"] = sil_samples_umap

# -----------------------------
# 5) Clustering sin reducción (MFCCs originales)
# -----------------------------

kmeans_raw = KMeans(n_clusters=K, n_init=50, random_state=SEMILLA)
labels_raw = kmeans_raw.fit_predict(X_std)
sil_raw = silhouette_score(X_std, labels_raw)
sil_samples_raw = silhouette_samples(X_std, labels_raw)

df_raw = df_filtrado.loc[idx].copy()
df_raw["cluster"] = labels_raw
df_raw["silhouette"] = sil_samples_raw

# -----------------------------
# 6) Gráficos estáticos (PCA y UMAP)
# -----------------------------

# PCA Scatter 2D
fig, ax = plt.subplots()
sns.scatterplot(
    data=df_pca, x="DIM1", y="DIM2",
    hue="cluster", palette="tab20", s=18, linewidth=0, alpha=0.85, ax=ax
)
ax.set_title(f"PCA(8) + K-means (k={K}) | sd1≥{UMBRAL_SD1} | VarExp2D={var_exp_2d:.2f}", fontsize=16)
ax.set_xlabel("Componente 1")
ax.set_ylabel("Componente 2")
ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig(ruta_salida / f"PA41_pca_scatter2d_k{K}.png")
plt.close(fig)

# UMAP Scatter 2D
fig, ax = plt.subplots()
sns.scatterplot(
    data=df_umap, x="DIM1", y="DIM2",
    hue="cluster", palette="tab20", s=18, linewidth=0, alpha=0.85, ax=ax
)
ax.set_title(f"UMAP(3D→2D) + K-means (k={K}) | nn={UMAP_N_NEIGHBORS}, md={UMAP_MIN_DIST}", fontsize=16)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig(ruta_salida / f"PA41_umap_scatter2d_k{K}.png")
plt.close(fig)

# Boxplot mfcc_mean_1 por cluster (PCA) — corregido warning
fig2, ax2 = plt.subplots()
sns.boxplot(
    data=df_pca, x="cluster", y="mfcc_mean_1",
    hue="cluster", palette="tab20", dodge=False, ax=ax2
)
ax2.legend_.remove()
ax2.set_title(f"Distribución de mfcc_mean_1 por cluster (PCA, k={K})")
plt.tight_layout()
plt.savefig(ruta_salida / f"PA41_pca_boxplot_mfcc_mean1_k{K}.png")
plt.close(fig2)

# Boxplot silhouette por cluster (PCA) — corregido warning
fig3, ax3 = plt.subplots()
sns.boxplot(
    data=df_pca, x="cluster", y="silhouette",
    hue="cluster", palette="tab20", dodge=False, ax=ax3
)
ax3.legend_.remove()
ax3.set_title(f"Silhouette por cluster (PCA, k={K})")
plt.tight_layout()
plt.savefig(ruta_salida / f"PA41_pca_boxplot_silhouette_k{K}.png")
plt.close(fig3)

# Boxplot silhouette por cluster (UMAP) — corregido warning
fig4, ax4 = plt.subplots()
sns.boxplot(
    data=df_umap, x="cluster", y="silhouette",
    hue="cluster", palette="tab20", dodge=False, ax=ax4
)
ax4.legend_.remove()
ax4.set_title(f"Silhouette por cluster (UMAP, k={K})")
plt.tight_layout()
plt.savefig(ruta_salida / f"PA41_umap_boxplot_silhouette_k{K}.png")
plt.close(fig4)

# -----------------------------
# 7) Gráficos interactivos 3D (Plotly)
# -----------------------------

# PCA 3D interactivo
hover_cols = [c for c in ["sitio", "archivo_origen", "tiempo_inicio", "tiempo_fin"] if c in df_pca.columns]
fig_pca3d = px.scatter_3d(
    df_pca,
    x="PC1", y="PC2", z="PC3",
    color="cluster",
    hover_data=hover_cols + ["mfcc_mean_1", "mfcc_sd_1"],
    title=f"PCA 3D (PC1–PC3) | k={K} | VarExp2D={var_exp_2d:.2f}"
)
fig_pca3d.update_traces(marker=dict(size=3, opacity=0.85))
fig_pca3d.write_html(str(ruta_salida / f"PA41_pca_scatter3d_k{K}.html"))

# UMAP 3D interactivo
fig_umap3d = px.scatter_3d(
    df_umap,
    x="U1", y="U2", z="U3",
    color="cluster",
    hover_data=hover_cols + ["mfcc_mean_1", "mfcc_sd_1"],
    title=f"UMAP 3D (U1–U3) | k={K} | nn={UMAP_N_NEIGHBORS}, md={UMAP_MIN_DIST}"
)
fig_umap3d.update_traces(marker=dict(size=3, opacity=0.85))
fig_umap3d.write_html(str(ruta_salida / f"PA41_umap_scatter3d_k{K}.html"))

# -----------------------------
# 8) Exportes de embeddings y trazabilidad
# -----------------------------

# PCA embedding
cols_export_pca = ["DIM1", "DIM2", "PC1", "PC2", "PC3", "cluster", "silhouette"] + \
    [c for c in FEATURES_BASE + FEATURES_SD if c in df_pca.columns] + \
    [c for c in ["sitio", "archivo_origen", "tiempo_inicio", "tiempo_fin"] if c in df_pca.columns]
df_pca[cols_export_pca].to_csv(ruta_salida / f"PA41_fase2_embedding_clusters_pca{N_COMPONENTES_PCA_CLUST}_k{K}.csv", index=False)

# UMAP embedding
cols_export_umap = ["DIM1", "DIM2", "U1", "U2", "U3", "cluster", "silhouette"] + \
    [c for c in FEATURES_BASE + FEATURES_SD if c in df_umap.columns] + \
    [c for c in ["sitio", "archivo_origen", "tiempo_inicio", "tiempo_fin"] if c in df_umap.columns]
df_umap[cols_export_umap].to_csv(ruta_salida / f"PA41_fase2_embedding_clusters_umap_k{K}.csv", index=False)

# RAW (sin reducción)
df_raw.to_csv(ruta_salida / f"PA41_fase2_embedding_clusters_sin_pca_k{K}.csv", index=False)

# Resumen por cluster (PCA)
cols_resumen = {
    "conteo": ("cluster", "size"),
    "mean1": ("mfcc_mean_1", "mean"),
    "mean2": ("mfcc_mean_2", "mean"),
    "mean3": ("mfcc_mean_3", "mean"),
    "silhouette_prom": ("silhouette", "mean"),
}
if "mfcc_sd_1" in df_pca.columns:
    cols_resumen["sd1_prom"] = ("mfcc_sd_1", "mean")

resumen_pca = df_pca.groupby("cluster").agg(**cols_resumen)
resumen_pca.to_csv(ruta_salida / f"PA41_pca_resumen_por_cluster_k{K}.csv")

# Resumen por cluster (UMAP)
resumen_umap = df_umap.groupby("cluster").agg(**cols_resumen)
resumen_umap.to_csv(ruta_salida / f"PA41_umap_resumen_por_cluster_k{K}.csv")

# -----------------------------
# 9) Comparativa de silhouette
# -----------------------------

df_sil = pd.DataFrame([
    {"metodo": "PCA", "k": K, "silhouette": sil_pca, "var_exp_total": var_exp_total, "var_exp_2d": var_exp_2d},
    {"metodo": "UMAP", "k": K, "silhouette": sil_umap, "var_exp_total": None, "var_exp_2d": None},
    {"metodo": "MFCCs originales", "k": K, "silhouette": sil_raw, "var_exp_total": None, "var_exp_2d": None},
])
df_sil.to_csv(ruta_salida / "PA41_fase2_silhouette_scores_comparativo.csv", index=False)

print("✅ Fase 2 extendida (PCA + UMAP + Plotly) completada.")
print(f"- Silhouette PCA: {sil_pca:.3f}")
print(f"- Silhouette UMAP: {sil_umap:.3f}")
print(f"- Silhouette sin reducción: {sil_raw:.3f}")
print("- Gráficos estáticos y HTML interactivos guardados en 'resultados'")
