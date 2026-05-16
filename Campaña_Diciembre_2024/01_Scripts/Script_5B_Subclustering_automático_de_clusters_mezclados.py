# ============================================================
# Script 5B — Subclustering menos fragmentado de clusters mezclados
# ------------------------------------------------------------
# Versión adaptada para usar las métricas de 5C:
#   - Lee *_metricas_5C.csv por sitio.
#   - Toma los clusters donde es_mezclado_recomendado == True
#     y cluster_hdbscan != -1.
#   - Para cada uno:
#       * hace UMAP local 2D + HDBSCAN (parámetros conservadores),
#       * genera figura 2D,
#       * guarda CSV con subclusters.
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import hdbscan

# ----------------- PARÁMETROS AJUSTABLES --------------------

NOMBRE_CAMPANIA = "Campaña diciembre 2024"
SUFIJO_CORRIDA = "v2_horario18a06_insectos6a12"

BASE_DIR = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local")
BASE_RESULTADOS = BASE_DIR / "resultados_HDD_Seagate" / NOMBRE_CAMPANIA
RUTA_PROYECCION_BD = BASE_DIR / "proyeccion_BD_fase1_5"

# Umbral de confianza para BD (en UMAP 2D):
UMBRAL_DIST_CONF = 1.0            # solo se grafican BD con dist_nn_umap2d <= este valor

# Parámetros de HDBSCAN MÁS CONSERVADORES (menos fragmentación)
HDBSCAN_MIN_CLUSTER_SIZE = 40
HDBSCAN_MIN_SAMPLES = 10

sns.set_theme(context="talk", style="whitegrid", palette="tab20")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["savefig.dpi"] = 200

# ------------------------------------------------------------

def detectar_clusters_mezclados_sitio(sitio: str):
    """
    Nueva versión:
      - Lee las métricas de 5C para el sitio.
      - Selecciona los clusters donde:
          * es_mezclado_recomendado == True
          * cluster_hdbscan != -1  (ruido de HDBSCAN)
    Devuelve una lista de cluster_ids a subclusterizar.
    """
    ruta_metricas_5c = BASE_RESULTADOS / sitio / f"{sitio}_{SUFIJO_CORRIDA}_metricas_5C.csv"
    if not ruta_metricas_5c.exists():
        print(f"⚠️ No se encontró métricas 5C para {sitio}: {ruta_metricas_5c}")
        return []

    df_m = pd.read_csv(ruta_metricas_5c)

    # Aseguramos nombres esperados
    cols_necesarias = {"cluster_hdbscan", "es_mezclado_recomendado"}
    if not cols_necesarias.issubset(df_m.columns):
        print(f"⚠️ Faltan columnas en métricas 5C para {sitio}. Se esperaban: {cols_necesarias}")
        return []

    # Filtrar clusters mezclados según 5C, excluyendo ruido (-1)
    df_sel = df_m[
        (df_m["es_mezclado_recomendado"] == True) &
        (df_m["cluster_hdbscan"] != -1)
    ]

    clusters_mezclados = sorted(df_sel["cluster_hdbscan"].unique().tolist())
    return clusters_mezclados


def subclustering_cluster_sitio(sitio: str, cluster_id: int):
    """
    Para un sitio y un cluster específico:
      - Toma los puntos de campaña de ese cluster (Fase 2).
      - Hace un UMAP local 2D + HDBSCAN (subclustering) con
        parámetros más conservadores.
      - Asigna subcluster a cada punto de campaña.
      - Genera figura 2D con:
          * campaña coloreada por subcluster
          * BD confiable (distancia <= UMBRAL_DIST_CONF) sobrepuesta
      - Guarda CSV con subclusters.
    """
    print(f"   → Subclustering (menos fragmentado) sitio {sitio}, cluster {cluster_id}")

    ruta_sitio = BASE_RESULTADOS / sitio
    ruta_fase2 = ruta_sitio / f"{sitio}_{SUFIJO_CORRIDA}_fase2_umap_hdbscan.csv"
    ruta_bd_detalle = RUTA_PROYECCION_BD / f"{sitio}_{SUFIJO_CORRIDA}_BD_con_clusters.csv"

    if not ruta_fase2.exists():
        print(f"     ⚠️ No se encontró Fase 2 para {sitio}: {ruta_fase2}")
        return
    if not ruta_bd_detalle.exists():
        print(f"     ⚠️ No se encontró BD_con_clusters para {sitio}: {ruta_bd_detalle}")
        return

    df_f2 = pd.read_csv(ruta_fase2)
    df_bd = pd.read_csv(ruta_bd_detalle)

    if "cluster_hdbscan" not in df_f2.columns:
        print(f"     ⚠️ No se encontró columna 'cluster_hdbscan' en Fase 2 para {sitio}")
        return

    # Filtrar puntos de campaña del cluster objetivo
    df_cluster = df_f2[df_f2["cluster_hdbscan"] == cluster_id].copy()
    if df_cluster.empty:
        print(f"     ⚠️ Cluster {cluster_id} vacío en campaña para {sitio}")
        return

    # UMAP local sobre U1, U2, U3 (representación global ya existente)
    cols_umap = ["U1", "U2", "U3"]
    for c in cols_umap:
        if c not in df_cluster.columns:
            print(f"     ⚠️ No se encontró columna {c} en Fase 2 para {sitio}")
            return

    X = df_cluster[cols_umap].values

    umap_local = umap.UMAP(
        n_neighbors=20,
        min_dist=0.15,
        n_components=2,
        metric="euclidean",
        random_state=42,
    )
    X_local_2d = umap_local.fit_transform(X)

    # HDBSCAN local con parámetros más conservadores
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    labels_local = clusterer.fit_predict(X_local_2d)

    df_cluster["subcluster_id"] = labels_local
    df_cluster["subU1"] = X_local_2d[:, 0]
    df_cluster["subU2"] = X_local_2d[:, 1]

    # Guardar CSV con subclusters
    ruta_subclusters_csv = ruta_sitio / f"{sitio}_{SUFIJO_CORRIDA}_cluster{cluster_id}_subclustering_5B.csv"
    df_cluster.to_csv(ruta_subclusters_csv, index=False)
    print(f"     💾 Subclusters (5B) guardados en: {ruta_subclusters_csv}")

    # BD confiable para graficar:
    if {"cluster_nn", "dist_nn_umap2d", "BD_DIM1", "BD_DIM2"}.issubset(df_bd.columns):
        df_bd_conf = df_bd[
            (df_bd["cluster_nn"] == cluster_id)
            & (df_bd["dist_nn_umap2d"] <= UMBRAL_DIST_CONF)
        ].copy()
    else:
        df_bd_conf = pd.DataFrame()
        print("     ⚠️ Columnas de BD incompletas, no se graficará BD confiable.")

    figuras_dir = ruta_sitio / "figuras_subclustering_5B"
    figuras_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(top=0.9, bottom=0.1)

    # Campaña: puntos coloreados por subcluster
    sns.scatterplot(
        data=df_cluster,
        x="subU1", y="subU2",
        hue="subcluster_id",
        palette="tab20",
        s=18, linewidth=0, alpha=0.7,
        ax=ax,
        legend=True
    )

    # BD confiable: puntos negros
    if not df_bd_conf.empty:
        ax.scatter(
            df_bd_conf["BD_DIM1"], df_bd_conf["BD_DIM2"],
            c="black", s=45, alpha=0.9,
            label="BD confiable", edgecolors="white", linewidths=0.3
        )

    ax.set_title(f"{sitio} | Cluster {cluster_id} — Subclustering 5B (menos fragmentado) + BD confiable")
    ax.set_xlabel("subU1 (UMAP local)")
    ax.set_ylabel("subU2 (UMAP local)")
    ax.legend(loc="best")
    plt.tight_layout()

    ruta_fig = figuras_dir / f"{sitio}_{SUFIJO_CORRIDA}_cluster{cluster_id}_subclustering_5B_2d.png"
    fig.savefig(ruta_fig)
    plt.close(fig)
    print(f"     💾 Figura 2D subclustering 5B guardada en: {ruta_fig}")


def procesar_sitio(sitio: str):
    print(f"\n=== Subclustering 5B (menos fragmentado) para sitio: {sitio} ===")

    clusters_mezclados = detectar_clusters_mezclados_sitio(sitio)
    if not clusters_mezclados:
        print("   No se detectaron clusters mezclados según métricas 5C.")
        return

    print(f"   Clusters mezclados detectados (5C): {clusters_mezclados}")

    for cluster_id in clusters_mezclados:
        subclustering_cluster_sitio(sitio, cluster_id)


def main():
    if not BASE_RESULTADOS.exists():
        print(f"⚠️ Carpeta de resultados no existe: {BASE_RESULTADOS}")
        return

    grabadoras = [p.name for p in BASE_RESULTADOS.iterdir() if p.is_dir()]
    grabadoras = sorted(grabadoras)

    print("Sitios detectados para subclustering 5B (menos fragmentado):")
    for g in grabadoras:
        print(" -", g)

    for sitio in grabadoras:
        procesar_sitio(sitio)

    print("\n✅ Subclustering 5B (menos fragmentado) completado.")


if __name__ == "__main__":
    main()
