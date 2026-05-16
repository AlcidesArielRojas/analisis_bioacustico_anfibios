# ============================================================
# Script 5B-hier — Subclustering jerárquico con corte automático
# ------------------------------------------------------------
# Este script:
#   1) Usa las métricas de 5C para decidir qué clusters son
#      candidatos a subclustering (es_mezclado_recomendado == True).
#   2) Para cada cluster candidato:
#        - toma U1, U2, U3 de Fase 2,
#        - aplica AgglomerativeClustering (jerárquico, linkage="ward"),
#        - prueba k = 2..K_MAX subclusters,
#        - elige k óptimo por silhouette score,
#        - si el mejor silhouette < UMBRAL_SILHOUETTE → no subclusteriza.
#   3) Guarda:
#        - CSV con subcluster_hier_id por punto de campaña,
#        - figura 2D (U1 vs U2) coloreada por subcluster.
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ----------------- PARÁMETROS AJUSTABLES --------------------

NOMBRE_CAMPANIA = "Campaña diciembre 2024"
SUFIJO_CORRIDA = "v2_horario18a06_insectos6a12"

BASE_DIR = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local")
BASE_RESULTADOS = BASE_DIR / "resultados_HDD_Seagate" / NOMBRE_CAMPANIA

# Número máximo de subclusters a considerar
K_MAX = 5

# Umbral mínimo de silhouette para aceptar subclustering
UMBRAL_SILHOUETTE = 0.12  # ajustable

# Tamaño mínimo del cluster original para intentar subclustering
MIN_SEGMENTOS_CLUSTER = 200  # por debajo de esto, muchas veces no vale la pena

sns.set_theme(context="talk", style="whitegrid", palette="tab20")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["savefig.dpi"] = 200

# ------------------------------------------------------------

def obtener_clusters_candidatos_sitio(sitio: str):
    """
    Lee las métricas de 5C para el sitio y devuelve
    los cluster_hdbscan que son candidatos a subclustering:
      - es_mezclado_recomendado == True
      - cluster_hdbscan != -1
      - n_segmentos >= MIN_SEGMENTOS_CLUSTER
    """
    ruta_metricas_5c = BASE_RESULTADOS / sitio / f"{sitio}_{SUFIJO_CORRIDA}_metricas_5C.csv"
    if not ruta_metricas_5c.exists():
        print(f"⚠️ No se encontró métricas 5C para {sitio}: {ruta_metricas_5c}")
        return []

    df_m = pd.read_csv(ruta_metricas_5c)

    cols_necesarias = {"cluster_hdbscan", "es_mezclado_recomendado", "n_segmentos"}
    if not cols_necesarias.issubset(df_m.columns):
        print(f"⚠️ Faltan columnas en métricas 5C para {sitio}. Se esperaban: {cols_necesarias}")
        return []

    df_sel = df_m[
        (df_m["es_mezclado_recomendado"] == True) &
        (df_m["cluster_hdbscan"] != -1) &
        (df_m["n_segmentos"] >= MIN_SEGMENTOS_CLUSTER)
    ]

    return sorted(df_sel["cluster_hdbscan"].unique().tolist())


def subclustering_jerarquico_cluster(sitio: str, cluster_id: int):
    """
    Para un sitio y un cluster específico:
      - Toma los puntos de campaña de ese cluster (Fase 2).
      - Usa U1, U2, U3 como features.
      - Prueba AgglomerativeClustering con k = 2..K_MAX.
      - Elige k óptimo por silhouette score.
      - Si el mejor silhouette < UMBRAL_SILHOUETTE → no subclusteriza (todo subcluster_hier_id = 0).
      - Si el mejor silhouette >= UMBRAL_SILHOUETTE → asigna subclusters jerárquicos.
      - Guarda CSV y figura 2D (U1 vs U2).
    """
    print(f"   → Subclustering jerárquico sitio {sitio}, cluster {cluster_id}")

    ruta_sitio = BASE_RESULTADOS / sitio
    ruta_fase2 = ruta_sitio / f"{sitio}_{SUFIJO_CORRIDA}_fase2_umap_hdbscan.csv"

    if not ruta_fase2.exists():
        print(f"     ⚠️ No se encontró Fase 2 para {sitio}: {ruta_fase2}")
        return

    df_f2 = pd.read_csv(ruta_fase2)

    if "cluster_hdbscan" not in df_f2.columns:
        print(f"     ⚠️ No se encontró columna 'cluster_hdbscan' en Fase 2 para {sitio}")
        return

    df_cluster = df_f2[df_f2["cluster_hdbscan"] == cluster_id].copy()
    if df_cluster.empty:
        print(f"     ⚠️ Cluster {cluster_id} vacío en campaña para {sitio}")
        return

    # Verificar columnas U1, U2, U3
    cols_umap = ["U1", "U2", "U3"]
    for c in cols_umap:
        if c not in df_cluster.columns:
            print(f"     ⚠️ No se encontró columna {c} en Fase 2 para {sitio}")
            return

    X = df_cluster[cols_umap].values
    n_total = X.shape[0]

    if n_total < 2 * K_MAX:
        print(f"     ⚠️ Cluster {cluster_id} tiene pocos puntos ({n_total}), no se subclusteriza.")
        df_cluster["subcluster_hier_id"] = 0
        guardar_resultados_jerarquico(sitio, cluster_id, df_cluster)
        return

    # Probar k = 2..K_MAX y elegir por silhouette
    mejores_labels = None
    mejor_k = 1
    mejor_sil = -1.0

    for k in range(2, K_MAX + 1):
        try:
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage="ward",
                metric="euclidean"   # CORREGIDO: reemplaza affinity=
            )
            labels = model.fit_predict(X)

            if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= n_total:
                continue

            sil = silhouette_score(X, labels)

        except Exception as e:
            print(f"     ⚠️ Error calculando silhouette para k={k} en cluster {cluster_id}: {e}")
            continue

        if sil > mejor_sil:
            mejor_sil = sil
            mejor_k = k
            mejores_labels = labels

    if mejores_labels is None or mejor_sil < UMBRAL_SILHOUETTE:
        print(f"     ℹ️ Mejor silhouette = {mejor_sil:.3f} < {UMBRAL_SILHOUETTE}. No se subclusteriza (todo id=0).")
        df_cluster["subcluster_hier_id"] = 0
    else:
        print(f"     ✅ Mejor k = {mejor_k} con silhouette = {mejor_sil:.3f}")
        df_cluster["subcluster_hier_id"] = mejores_labels

    guardar_resultados_jerarquico(sitio, cluster_id, df_cluster)


def guardar_resultados_jerarquico(sitio: str, cluster_id: int, df_cluster: pd.DataFrame):
    """
    Guarda:
      - CSV con subcluster_hier_id
      - Figura 2D U1 vs U2 coloreada por subcluster_hier_id
    """
    ruta_sitio = BASE_RESULTADOS / sitio

    # CSV
    ruta_csv = ruta_sitio / f"{sitio}_{SUFIJO_CORRIDA}_cluster{cluster_id}_subclustering_hier_5B.csv"
    df_cluster.to_csv(ruta_csv, index=False)
    print(f"     💾 Subclusters jerárquicos guardados en: {ruta_csv}")

    # Figura
    figuras_dir = ruta_sitio / "figuras_subclustering_hier_5B"
    figuras_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(top=0.9, bottom=0.1)

    sns.scatterplot(
        data=df_cluster,
        x="U1", y="U2",
        hue="subcluster_hier_id",
        palette="tab20",
        s=18, linewidth=0, alpha=0.7,
        ax=ax,
        legend=True
    )

    ax.set_title(f"{sitio} | Cluster {cluster_id} — Subclustering jerárquico 5B-hier (corte automático)")
    ax.set_xlabel("U1 (UMAP global)")
    ax.set_ylabel("U2 (UMAP global)")
    ax.legend(loc="best")
    plt.tight_layout()

    ruta_fig = figuras_dir / f"{sitio}_{SUFIJO_CORRIDA}_cluster{cluster_id}_subclustering_hier_5B_2d.png"
    fig.savefig(ruta_fig)
    plt.close(fig)
    print(f"     💾 Figura 2D subclustering jerárquico guardada en: {ruta_fig}")


def procesar_sitio(sitio: str):
    print(f"\n=== Subclustering jerárquico 5B-hier para sitio: {sitio} ===")

    clusters_candidatos = obtener_clusters_candidatos_sitio(sitio)
    if not clusters_candidatos:
        print("   No se detectaron clusters candidatos según métricas 5C.")
        return

    print(f"   Clusters candidatos (5C): {clusters_candidatos}")

    for cluster_id in clusters_candidatos:
        subclustering_jerarquico_cluster(sitio, cluster_id)


def main():
    if not BASE_RESULTADOS.exists():
        print(f"⚠️ Carpeta de resultados no existe: {BASE_RESULTADOS}")
        return

    sitios = [p.name for p in BASE_RESULTADOS.iterdir() if p.is_dir()]
    sitios = sorted(sitios)

    print("Sitios detectados para subclustering jerárquico 5B-hier:")
    for g in sitios:
        print(" -", g)

    for sitio in sitios:
        procesar_sitio(sitio)

    print("\n✅ Subclustering jerárquico 5B-hier completado.")


if __name__ == "__main__":
    main()
