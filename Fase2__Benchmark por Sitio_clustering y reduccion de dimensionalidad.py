# ================================================================
# Fase 2 extendida: CANTERA1Tapy
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
ruta_features = ruta_salida / "features_CANTERA1Tapy.parquet"
ruta_salida.mkdir(parents=True, exist_ok=True)

SEMILLA = 123
UMBRAL_SD1 = 12.0   # podés ajustar si querés incluir más segmentos
FRACCION_MUESTRA = 0.25
N_COMPONENTES_PCA_CLUST = 6
K = 6

# UMAP params
UMAP_N_NEIGHBORS = 14
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 3  # para 3D interactivo y también derivar 2D

# Selección de features (evitando redundancia fuerte entre mfcc_mean_2 y mfcc_mean_4)
FEATURES_BASE = ["mfcc_mean_1", "mfcc_mean_3", "mfcc_mean_5"]
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
ax.set_title(f"CANTERA PCA(8) + K-means (k={K}) | sd1≥{UMBRAL_SD1} | VarExp2D={var_exp_2d:.2f}", fontsize=16)
ax.set_xlabel("Componente 1")
ax.set_ylabel("Componente 2")
ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig(ruta_salida / f"CANTERA_pca_scatter2d_k{K}.png")
plt.close(fig)

# UMAP Scatter 2D
fig, ax = plt.subplots()
sns.scatterplot(
    data=df_umap, x="DIM1", y="DIM2",
    hue="cluster", palette="tab20", s=18, linewidth=0, alpha=0.85, ax=ax
)
ax.set_title(f"CANTERA UMAP(3D→2D) + K-means (k={K}) | nn={UMAP_N_NEIGHBORS}, md={UMAP_MIN_DIST}", fontsize=16)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig(ruta_salida / f"CANTERA_umap_scatter2d_k{K}.png")
plt.close(fig)

# Boxplot mfcc_mean_1 por cluster (PCA)
fig2, ax2 = plt.subplots()
sns.boxplot(
    data=df_pca, x="cluster", y="mfcc_mean_1",
    hue="cluster", palette="tab20", dodge=False, ax=ax2
)
ax2.legend_.remove()
ax2.set_title(f"CANTERA Distribución de mfcc_mean_1 por cluster (PCA, k={K})")
plt.tight_layout()
plt.savefig(ruta_salida / f"CANTERA_pca_boxplot_mfcc_mean1_k{K}.png")
plt.close(fig2)

# Boxplot silhouette por cluster (PCA)
fig3, ax3 = plt.subplots()
sns.boxplot(
    data=df_pca, x="cluster", y="silhouette",
    hue="cluster")
