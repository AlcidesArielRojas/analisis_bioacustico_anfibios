# ================================================================
# Fase 2 Benchmark (multi-sitio):
# UMAP + HDBSCAN + Selection Table para Raven
# ================================================================

from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import umap
import plotly.express as px
from hdbscan import HDBSCAN

BASE_DIR = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local")
ruta_salida = BASE_DIR / "resultados"
ruta_salida.mkdir(parents=True, exist_ok=True)

# Mismos sitios y sufijo que en Fase 1
SITIOS = [
    "BO-Tapyta",
]
SUFIJO_CORRIDA = "v2_horario18a06_insectos6a12"

SEMILLA = 123

UMAP_N_NEIGHBORS = 60
UMAP_MIN_DIST = 0.3
UMAP_N_COMPONENTS = 3
UMAP_METRIC = "cosine"
UMAP_DENSMAP = False

PCA_COMPONENTS = 20

MIN_CLUSTER_SIZE = 800
MIN_SAMPLES = 65
CLUSTER_SELECTION_EPS = 0.03
CLUSTER_SELECTION_METHOD = "eom"

sns.set_theme(context="talk", style="whitegrid", palette="tab20")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["savefig.dpi"] = 200

def correr_fase2_para_sitio(SITE: str):
    print(f"\n==============================")
    print(f"Fase 2 para sitio: {SITE}")
    print(f"==============================")

    figuras_dir = BASE_DIR / f"figuras_{SITE}_{SUFIJO_CORRIDA}"
    figuras_dir.mkdir(parents=True, exist_ok=True)

    ruta_features = ruta_salida / f"features_{SITE}_{SUFIJO_CORRIDA}.parquet"
    if not ruta_features.exists():
        print(f"⚠️ No se encontró features para {SITE}: {ruta_features}")
        return

    df = pd.read_parquet(ruta_features)
    FEATURES_BASE = [c for c in df.columns if c.startswith("mfcc_mean_")]
    FEATURES_SD   = [c for c in df.columns if c.startswith("mfcc_sd_")]

    features = FEATURES_BASE + FEATURES_SD
    if not features:
        print(f"⚠️ No se encontraron columnas MFCC en {ruta_features}")
        return

    df_muestra = df.copy()
    X = df_muestra[features].dropna()

    # Submuestreo para evitar MemoryError
    MAX_SEGMENTOS = 100_000  # ajustable según tu RAM
    if len(X) > MAX_SEGMENTOS:
        print(f"⚠️ Demasiados segmentos ({len(X)}), se submuestrean a {MAX_SEGMENTOS}")
        X = X.sample(n=MAX_SEGMENTOS, random_state=42)
        idx = X.index
        df_muestra = df_muestra.loc[idx].copy()

    idx = X.index

    scaler = RobustScaler()
    X_std = scaler.fit_transform(X)

    pca = PCA(n_components=PCA_COMPONENTS, whiten=True, random_state=SEMILLA)
    X_whiten = pca.fit_transform(X_std)

    umap_model = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    n_components=UMAP_N_COMPONENTS,
    metric=UMAP_METRIC,
    densmap=UMAP_DENSMAP,
    random_state=None,
    n_jobs=-1)

    X_umap = umap_model.fit_transform(X_whiten)

    df_umap = df_muestra.loc[idx].copy()
    df_umap["DIM1"] = X_umap[:, 0]
    df_umap["DIM2"] = X_umap[:, 1]
    df_umap["U1"] = X_umap[:, 0]
    df_umap["U2"] = X_umap[:, 1]
    df_umap["U3"] = X_umap[:, 2] if UMAP_N_COMPONENTS >= 3 else 0

    hdb = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        cluster_selection_epsilon=CLUSTER_SELECTION_EPS,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
        prediction_data=True
    )
    labels_hdb = hdb.fit_predict(X_umap)

    df_umap["cluster_hdbscan"] = labels_hdb
    df_umap["es_ruido"] = (labels_hdb == -1).astype(int)

    mask_clusters = labels_hdb != -1
    n_clusters = len(set(labels_hdb[mask_clusters]))
    sil_hdb = silhouette_score(X_umap[mask_clusters], labels_hdb[mask_clusters]) if n_clusters > 1 else np.nan
    dbi_hdb = davies_bouldin_score(X_umap[mask_clusters], labels_hdb[mask_clusters]) if n_clusters > 1 else np.nan
    ch_hdb = calinski_harabasz_score(X_umap[mask_clusters], labels_hdb[mask_clusters]) if n_clusters > 1 else np.nan
    proporcion_ruido = 1.0 - (mask_clusters.sum() / len(labels_hdb))

    fig, ax = plt.subplots(figsize=(16, 12))
    plt.subplots_adjust(top=0.88, bottom=0.12)
    sns.scatterplot(
        data=df_umap, x="DIM1", y="DIM2",
        hue="cluster_hdbscan", palette="tab20", s=18, linewidth=0, alpha=0.85, ax=ax
    )
    ax.set_title(f"{SITE} | UMAP + HDBSCAN (ruido=-1) | clusters={n_clusters} | ruido={proporcion_ruido:.2f}")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(ruta_salida / f"{SITE}_{SUFIJO_CORRIDA}_umap_hdbscan_scatter2d.png")
    plt.savefig(figuras_dir / f"{SITE}_{SUFIJO_CORRIDA}_umap_hdbscan_scatter2d.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 12))
    plt.subplots_adjust(top=0.88, bottom=0.12)
    sns.scatterplot(
        data=df_umap, x="DIM1", y="DIM2",
        hue="es_ruido", palette={0: "tab:blue", 1: "tab:red"}, s=18, linewidth=0, alpha=0.85, ax=ax
    )
    ax.set_title(f"{SITE} | UMAP puntos válidos (azul) vs ruido (rojo)")
    ax.legend(title="Ruido", labels=["No ruido (0)", "Ruido (1)"])
    plt.tight_layout()
    plt.savefig(ruta_salida / f"{SITE}_{SUFIJO_CORRIDA}_umap_ruido_vs_valido.png")
    plt.savefig(figuras_dir / f"{SITE}_{SUFIJO_CORRIDA}_umap_ruido_vs_valido.png")
    plt.close(fig)

    hover_cols = [c for c in ["sitio", "archivo_origen", "tiempo_inicio", "tiempo_fin"] if c in df_umap.columns]
    fig_umap3d = px.scatter_3d(
        df_umap, x="U1", y="U2", z="U3",
        color="cluster_hdbscan",
        hover_data=hover_cols + [c for c in ["mfcc_mean_1", "mfcc_sd_1"] if c in df_umap.columns],
        title=f"{SITE} UMAP 3D + HDBSCAN | clusters={n_clusters} | ruido={proporcion_ruido:.2f}"
    )
    fig_umap3d.update_traces(marker=dict(size=3, opacity=0.85))
    fig_umap3d.write_html(str(ruta_salida / f"{SITE}_{SUFIJO_CORRIDA}_umap3d_hdbscan.html"))
    fig_umap3d.write_html(str(figuras_dir / f"{SITE}_{SUFIJO_CORRIDA}_umap3d_hdbscan.html"))

    cols_export = ["DIM1", "DIM2", "U1", "U2", "U3", "cluster_hdbscan", "es_ruido"] + \
                  [c for c in FEATURES_BASE + FEATURES_SD if c in df_umap.columns] + \
                  hover_cols
    df_umap[cols_export].to_csv(ruta_salida / f"{SITE}_{SUFIJO_CORRIDA}_fase2_umap_hdbscan.csv", index=False)

    df_metrics = pd.DataFrame([{
        "site": SITE,
        "n_clusters": n_clusters,
        "proporcion_ruido": proporcion_ruido,
        "silhouette": sil_hdb,
        "davies_bouldin": dbi_hdb,
        "calinski_harabasz": ch_hdb,
        "umap_n_neighbors": UMAP_N_NEIGHBORS,
        "umap_min_dist": UMAP_MIN_DIST,
        "umap_metric": UMAP_METRIC,
        "umap_densmap": UMAP_DENSMAP,
        "pca_components": PCA_COMPONENTS,
        "pca_whiten": True,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "min_samples": MIN_SAMPLES,
        "cluster_selection_epsilon": CLUSTER_SELECTION_EPS,
        "cluster_selection_method": CLUSTER_SELECTION_METHOD
    }])
    df_metrics.to_csv(ruta_salida / f"{SITE}_{SUFIJO_CORRIDA}_fase2_umap_hdbscan_metrics.csv", index=False)

    # Selection Table para Raven
    if all(col in df_umap.columns for col in ["archivo_origen", "tiempo_inicio", "tiempo_fin", "cluster_hdbscan"]):
        df_raven = df_umap[df_umap["cluster_hdbscan"] != -1].copy()
        df_raven = df_raven.reset_index(drop=True)
        df_raven["Selection"] = df_raven.index + 1
        df_raven["View"] = 1
        df_raven["Channel"] = 1
        df_raven["Begin Time (s)"] = df_raven["tiempo_inicio"]
        df_raven["End Time (s)"] = df_raven["tiempo_fin"]
        df_raven["Low Freq (Hz)"] = 0.0
        df_raven["High Freq (Hz)"] = 8000.0
        df_raven["Annotation"] = df_raven["cluster_hdbscan"].apply(lambda c: f"cluster_{c}")

        cols_raven = [
            "Selection", "View", "Channel",
            "Begin Time (s)", "End Time (s)",
            "Low Freq (Hz)", "High Freq (Hz)",
            "Annotation", "archivo_origen", "sitio"
        ]
        cols_raven = [c for c in cols_raven if c in df_raven.columns]

        ruta_raven = ruta_salida / f"{SITE}_{SUFIJO_CORRIDA}_fase2_raven_selection_table.txt"
        df_raven[cols_raven].to_csv(ruta_raven, sep="\t", index=False)
        print(f"📄 Selection Table para Raven guardada en: {ruta_raven}")
    else:
        print("⚠️ No se pudo generar Selection Table para Raven (faltan columnas).")

    print("✅ Fase 2 completada.")
    print(f"- Clusters detectados (sin ruido): {n_clusters}")
    print(f"- Proporción de ruido: {proporcion_ruido:.2f}")
    print(f"- Silhouette (sin ruido): {sil_hdb if not np.isnan(sil_hdb) else 'N/A'}")
    print(f"- Davies-Bouldin (menor es mejor): {dbi_hdb if not np.isnan(dbi_hdb) else 'N/A'}")
    print(f"- Calinski-Harabasz (mayor es mejor): {ch_hdb if not np.isnan(ch_hdb) else 'N/A'}")

if __name__ == "__main__":
    for sitio in SITIOS:
        correr_fase2_para_sitio(sitio)