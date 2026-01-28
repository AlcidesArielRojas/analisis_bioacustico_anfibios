# ================================================================
# Fase 2: CANTERA1Tapy (UMAP + HDBSCAN, sin filtro de banda)
# Embeddings UMAP, clustering HDBSCAN, métricas y trazabilidad
# ================================================================

from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import umap
import plotly.express as px
from hdbscan import HDBSCAN

# -----------------------------
# Configuración general
# -----------------------------
BASE_DIR = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local")
SITE = "CANTERA1Tapy"
DATA_SUBDIR = "Data"

ruta_salida = BASE_DIR / "resultados"
ruta_salida.mkdir(parents=True, exist_ok=True)

figuras_dir = BASE_DIR / f"figuras_{SITE}"
figuras_dir.mkdir(parents=True, exist_ok=True)

ruta_features = ruta_salida / f"features_{SITE}.parquet"
if not ruta_features.exists():
    posibles = list(ruta_salida.glob(f"*{SITE}*.parquet"))
    if posibles:
        ruta_features = posibles[0]
    else:
        raise FileNotFoundError(f"No se encontró features para {SITE}")

SEMILLA = 123
FRACCION_MUESTRA = 1.0

# UMAP parámetros (ajustables)
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 3  # 3D interactivo y derivar 2D

# HDBSCAN parámetros (ajustables)
MIN_CLUSTER_SIZE = 60
MIN_SAMPLES = 10

# Selección de features
FEATURES_BASE = ["mfcc_mean_1", "mfcc_mean_3", "mfcc_mean_5"]
FEATURES_SD = ["mfcc_sd_1", "mfcc_sd_2", "mfcc_sd_3"]

sns.set_theme(context="talk", style="whitegrid", palette="tab20")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["savefig.dpi"] = 200

# -----------------------------
# 1) Carga, muestreo y escalado
# -----------------------------
df = pd.read_parquet(ruta_features)
features = [f for f in FEATURES_BASE + FEATURES_SD if f in df.columns]
if not features:
    raise ValueError("No se encontraron columnas MFCC esperadas en el parquet.")

df_muestra = df.sample(frac=FRACCION_MUESTRA, random_state=SEMILLA)
X = df_muestra[features].dropna()
idx = X.index

scaler = RobustScaler()
X_std = scaler.fit_transform(X)

# -----------------------------
# 2) Embedding UMAP (2D/3D)
# -----------------------------
umap_model = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    n_components=UMAP_N_COMPONENTS,
    random_state=SEMILLA
)
X_umap = umap_model.fit_transform(X_std)

df_umap = df_muestra.loc[idx].copy()
df_umap["DIM1"] = X_umap[:, 0]
df_umap["DIM2"] = X_umap[:, 1]
df_umap["U1"] = X_umap[:, 0]
df_umap["U2"] = X_umap[:, 1]
df_umap["U3"] = X_umap[:, 2] if UMAP_N_COMPONENTS >= 3 else 0

# -----------------------------
# 3) Clustering HDBSCAN
# -----------------------------
hdb = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
              min_samples=MIN_SAMPLES,
              cluster_selection_epsilon=0.0,
              prediction_data=True)
labels_hdb = hdb.fit_predict(X_umap)  # -1 es ruido

df_umap["cluster_hdbscan"] = labels_hdb
df_umap["es_ruido"] = (labels_hdb == -1).astype(int)

# Métricas internas (sobre puntos no ruido si hay ≥2 clusters)
mask_clusters = labels_hdb != -1
n_clusters = len(set(labels_hdb[mask_clusters]))
sil_hdb = silhouette_score(X_umap[mask_clusters], labels_hdb[mask_clusters]) if n_clusters > 1 else np.nan
dbi_hdb = davies_bouldin_score(X_umap[mask_clusters], labels_hdb[mask_clusters]) if n_clusters > 1 else np.nan
ch_hdb = calinski_harabasz_score(X_umap[mask_clusters], labels_hdb[mask_clusters]) if n_clusters > 1 else np.nan
proporcion_ruido = 1.0 - (mask_clusters.sum() / len(labels_hdb))

# -----------------------------
# 4) Visualizaciones
# -----------------------------
# UMAP 2D por cluster (muestra ruido=-1)
fig, ax = plt.subplots()
sns.scatterplot(
    data=df_umap, x="DIM1", y="DIM2",
    hue="cluster_hdbscan", palette="tab20", s=18, linewidth=0, alpha=0.85, ax=ax
)
ax.set_title(f"{SITE} | UMAP + HDBSCAN (ruido=-1) | clusters={n_clusters} | ruido={proporcion_ruido:.2f}")
ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig(ruta_salida / f"{SITE}_umap_hdbscan_scatter2d.png")
plt.savefig(figuras_dir / f"{SITE}_umap_hdbscan_scatter2d.png")
plt.close(fig)

# UMAP 2D ruido vs no-ruido
fig, ax = plt.subplots()
sns.scatterplot(
    data=df_umap, x="DIM1", y="DIM2",
    hue="es_ruido", palette={0: "tab:blue", 1: "tab:red"}, s=18, linewidth=0, alpha=0.85, ax=ax
)
ax.set_title(f"{SITE} | UMAP puntos válidos (azul) vs ruido (rojo)")
ax.legend(title="Ruido", labels=["No ruido (0)", "Ruido (1)"])
plt.tight_layout()
plt.savefig(ruta_salida / f"{SITE}_umap_ruido_vs_valido.png")
plt.savefig(figuras_dir / f"{SITE}_umap_ruido_vs_valido.png")
plt.close(fig)

# UMAP 3D interactivo con hover
hover_cols = [c for c in ["sitio", "archivo_origen", "tiempo_inicio", "tiempo_fin"] if c in df_umap.columns]
fig_umap3d = px.scatter_3d(
    df_umap, x="U1", y="U2", z="U3",
    color="cluster_hdbscan",
    hover_data=hover_cols + [c for c in ["mfcc_mean_1", "mfcc_sd_1"] if c in df_umap.columns],
    title=f"{SITE} UMAP 3D + HDBSCAN | clusters={n_clusters} | ruido={proporcion_ruido:.2f}"
)
fig_umap3d.update_traces(marker=dict(size=3, opacity=0.85))
fig_umap3d.write_html(str(ruta_salida / f"{SITE}_umap3d_hdbscan.html"))
fig_umap3d.write_html(str(figuras_dir / f"{SITE}_umap3d_hdbscan.html"))

# -----------------------------
# 5) Exportes y métricas
# -----------------------------
cols_export = ["DIM1", "DIM2", "U1", "U2", "U3", "cluster_hdbscan", "es_ruido"] + \
              [c for c in FEATURES_BASE + FEATURES_SD if c in df_umap.columns] + \
              hover_cols
df_umap[cols_export].to_csv(ruta_salida / f"{SITE}_fase2_umap_hdbscan.csv", index=False)

df_metrics = pd.DataFrame([{
    "site": SITE,
    "n_clusters": n_clusters,
    "proporcion_ruido": proporcion_ruido,
    "silhouette": sil_hdb,
    "davies_bouldin": dbi_hdb,
    "calinski_harabasz": ch_hdb,
    "umap_n_neighbors": UMAP_N_NEIGHBORS,
    "umap_min_dist": UMAP_MIN_DIST,
    "min_cluster_size": MIN_CLUSTER_SIZE,
    "min_samples": MIN_SAMPLES
}])
df_metrics.to_csv(ruta_salida / f"{SITE}_fase2_umap_hdbscan_metrics.csv", index=False)

# Copias clave al directorio de figuras del sitio
for fname in [
    f"{SITE}_umap_hdbscan_scatter2d.png",
    f"{SITE}_umap_ruido_vs_valido.png",
    f"{SITE}_umap3d_hdbscan.html",
    f"{SITE}_fase2_umap_hdbscan.csv",
    f"{SITE}_fase2_umap_hdbscan_metrics.csv",
]:
    src = ruta_salida / fname
    dst = figuras_dir / fname
    try:
        if src.exists():
            shutil.copyfile(src, dst)
    except Exception:
        pass

print("✅ Fase 2 (UMAP + HDBSCAN) completada.")
print(f"- Clusters detectados (sin ruido): {n_clusters}")
print(f"- Proporción de ruido: {proporcion_ruido:.2f}")
print(f"- Silhouette (sin ruido): {sil_hdb if not np.isnan(sil_hdb) else 'N/A'}")
print(f"- Davies-Bouldin (menor es mejor): {dbi_hdb if not np.isnan(dbi_hdb) else 'N/A'}")
print(f"- Calinski-Harabasz (mayor es mejor): {ch_hdb if not np.isnan(ch_hdb) else 'N/A'}")
print(f"- Resultados en: {ruta_salida} | Figuras en: {figuras_dir}")