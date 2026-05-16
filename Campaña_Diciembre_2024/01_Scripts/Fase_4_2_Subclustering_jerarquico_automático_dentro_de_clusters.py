# ================================================================
# Fase 4.2: Subclustering jerárquico automático dentro de clusters
# ------------------------------------------------
# Qué hace este script (en sencillo):
# - Lee el CSV de métricas de Fase 4.1 (clusters candidatos).
# - Para cada (sitio, cluster_hdbscan) marcado como candidato:
#     * carga el CSV de Fase 2 del sitio,
#     * toma solo las filas de ese cluster,
#     * si el cluster es "normal" (no muy grande):
#           - aplica AgglomerativeClustering sobre todos los puntos (U1, U2, U3),
#           - prueba varios números de subclusters y elige el mejor por silhouette,
#           - asigna un subcluster_id a cada segmento.
#     * si el cluster es "gigante":
#           - toma una muestra aleatoria (p.ej. 20 000 puntos),
#           - hace el subclustering solo en la muestra,
#           - calcula centroides de cada subcluster,
#           - asigna TODOS los puntos del cluster al subcluster más cercano
#             según distancia en UMAP.
# - Genera un CSV por sitio con una columna nueva 'subcluster_id'.
# ================================================================

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ---------------- CONFIGURACIÓN (MODIFICABLE) ----------------

NOMBRE_CAMPANIA = "Campaña diciembre 2024"
SUFIJO_CORRIDA = "v2_horario18a06_insectos6a12"

BASE_DIR = Path(r"C:/Users/User/Proyecto_Paisajes_Sonoros_Repositorio_Local")
BASE_RESULTADOS = BASE_DIR / "resultados_HDD_Seagate" / NOMBRE_CAMPANIA

# Debe coincidir con el archivo generado en Fase 4.1
RUTA_SALIDA_METRICAS = BASE_RESULTADOS / f"{NOMBRE_CAMPANIA}_{SUFIJO_CORRIDA}_fase4_clusters_metrics.csv"

# Rango de número de subclusters a probar (ej. 2 a 3)
RANGO_SUBCLUSTERS = range(2, 4)

# Mínimo de segmentos para considerar un subcluster válido
MIN_SEGMENTOS_POR_SUBCLUSTER = 50

# Tamaño máximo de muestra para subclustering en clusters gigantes
MAX_PUNTOS_MUESTRA_SUBCLUSTERING = 20000

# Límite duro opcional: si es None, no se usa; si es un entero, salta clusters con más puntos que ese valor
MAX_SEGMENTOS_PARA_SUBCLUSTERING = None  # por ejemplo 150000 si querés un seguro extra


# ---------------- FUNCIONES AUXILIARES ----------------

def elegir_mejor_subclustering_en_muestra(X_muestra: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Dado un array X_muestra (n_samples, 3) con U1, U2, U3 de la MUESTRA,
    prueba varios números de subclusters y devuelve:
      - labels_muestra: etiquetas de subcluster para cada punto de la muestra,
      - centroides: dict {label: vector_centroid (3,)} en el espacio UMAP.
    """
    mejor_score = -1.0
    mejor_labels = None

    for n_clust in RANGO_SUBCLUSTERS:
        if X_muestra.shape[0] <= n_clust:
            continue
        try:
            model = AgglomerativeClustering(
                n_clusters=n_clust,
                linkage="ward"
            )
            labels = model.fit_predict(X_muestra)

            # silhouette requiere al menos 2 clusters distintos
            if len(set(labels)) < 2:
                continue

            score = silhouette_score(X_muestra, labels)
        except Exception as e:
            print(f"   ⚠️ Error con n_clusters={n_clust}: {e}")
            continue

        if score > mejor_score:
            mejor_score = score
            mejor_labels = labels

    if mejor_labels is None:
        # fallback: todo en un solo subcluster (0)
        mejor_labels = np.zeros(X_muestra.shape[0], dtype=int)

    # Calcular centroides en la muestra
    centroides = {}
    for lab in sorted(set(mejor_labels)):
        coords_lab = X_muestra[mejor_labels == lab]
        centroides[lab] = coords_lab.mean(axis=0)

    return mejor_labels, centroides


def asignar_todos_por_centroides(X_todos: np.ndarray, centroides: dict) -> np.ndarray:
    """
    Asigna cada punto de X_todos al subcluster cuyo centroide esté más cerca
    (distancia euclídea en UMAP).
    centroides: dict {label: vector_centroid (3,)}
    Devuelve un array de etiquetas (mismo largo que X_todos).
    """
    labels_centroides = sorted(centroides.keys())
    C = np.vstack([centroides[lab] for lab in labels_centroides])  # (k, 3)

    # Distancias (n_puntos, k)
    # dist^2 = sum((x - c)^2) → usamos broadcasting
    X_exp = X_todos[:, np.newaxis, :]      # (n, 1, 3)
    C_exp = C[np.newaxis, :, :]           # (1, k, 3)
    dist2 = np.sum((X_exp - C_exp) ** 2, axis=2)  # (n, k)

    idx_min = np.argmin(dist2, axis=1)    # índice del centroide más cercano
    labels_asignados = np.array([labels_centroides[i] for i in idx_min], dtype=int)
    return labels_asignados


# ---------------- LÓGICA PRINCIPAL POR CLUSTER ----------------

def procesar_sitio_y_cluster(sitio: str, k_cluster: int):
    """
    Aplica subclustering al cluster k_cluster de un sitio.
    - Si el cluster es "normal": subclustering directo sobre todos los puntos.
    - Si es "gigante": subclustering en una muestra y asignación por centroides.
    Guarda un CSV actualizado con columna 'subcluster_id'.
    """
    ruta_csv = BASE_RESULTADOS / sitio / f"{sitio}_{SUFIJO_CORRIDA}_fase2_umap_hdbscan.csv"
    if not ruta_csv.exists():
        print(f"⚠️ No se encontró CSV de Fase 2 para {sitio}: {ruta_csv}")
        return

    df = pd.read_csv(ruta_csv)

    for col in ["U1", "U2", "U3", "cluster_hdbscan"]:
        if col not in df.columns:
            print(f"⚠️ {sitio}: falta columna {col} en {ruta_csv}")
            return

    # Inicializar subcluster_id en -1 (sin subcluster)
    if "subcluster_id" not in df.columns:
        df["subcluster_id"] = -1

    mask = df["cluster_hdbscan"] == k_cluster
    df_k = df[mask].copy()
    if df_k.empty:
        print(f"⚠️ {sitio} | cluster {k_cluster}: no hay filas.")
        return

    n_seg = len(df_k)
    if n_seg < 2:
        print(f"⚠️ {sitio} | cluster {k_cluster}: muy pocos puntos para subclustering.")
        return

    # Límite duro opcional
    if (MAX_SEGMENTOS_PARA_SUBCLUSTERING is not None) and (n_seg > MAX_SEGMENTOS_PARA_SUBCLUSTERING):
        print(f"   ⚠️ {sitio} | cluster {k_cluster}: tiene {n_seg} segmentos, se salta por tamaño (> {MAX_SEGMENTOS_PARA_SUBCLUSTERING}).")
        return

    X_todos = df_k[["U1", "U2", "U3"]].values

    print(f"   → Subclustering {sitio} | cluster {k_cluster} con {n_seg} segmentos...")

    # Caso 1: cluster "normal" (no muy grande) → subclustering directo
    if n_seg <= MAX_PUNTOS_MUESTRA_SUBCLUSTERING:
        labels_muestra, centroides = elegir_mejor_subclustering_en_muestra(X_todos)
        labels_todos = labels_muestra  # usamos directamente las etiquetas
    else:
        # Caso 2: cluster gigante → subclustering en muestra + asignación por centroides
        print(f"   → Cluster grande, usando muestra de {MAX_PUNTOS_MUESTRA_SUBCLUSTERING} puntos para definir subclusters.")
        idx_muestra = np.random.choice(n_seg, size=MAX_PUNTOS_MUESTRA_SUBCLUSTERING, replace=False)
        X_muestra = X_todos[idx_muestra]

        labels_muestra, centroides = elegir_mejor_subclustering_en_muestra(X_muestra)
        labels_todos = asignar_todos_por_centroides(X_todos, centroides)

    # Verificar tamaño mínimo de subclusters
    df_k["subcluster_id_temp"] = labels_todos
    counts = df_k["subcluster_id_temp"].value_counts()
    subclusters_validos = counts[counts >= MIN_SEGMENTOS_POR_SUBCLUSTER].index.tolist()

    if not subclusters_validos:
        print(f"   ⚠️ {sitio} | cluster {k_cluster}: ningún subcluster supera el mínimo de {MIN_SEGMENTOS_POR_SUBCLUSTER} segmentos.")
        return

    # Re-etiquetar subclusters válidos a 0..M-1
    mapa_sub = {old: new for new, old in enumerate(sorted(subclusters_validos))}
    df_k["subcluster_id"] = df_k["subcluster_id_temp"].map(mapa_sub).fillna(-1).astype(int)
    df_k = df_k.drop(columns=["subcluster_id_temp"])

    # Actualizar df original
    df.loc[mask, "subcluster_id"] = df_k["subcluster_id"].values

    # Guardar CSV actualizado (nuevo archivo para Fase 4)
    ruta_out = BASE_RESULTADOS / sitio / f"{sitio}_{SUFIJO_CORRIDA}_fase4_umap_hdbscan_subclusters.csv"
    df.to_csv(ruta_out, index=False)
    print(f"   ✅ CSV con subclusters guardado en: {ruta_out}")


# ---------------- BUCLE PRINCIPAL ----------------

def main():
    if not RUTA_SALIDA_METRICAS.exists():
        print(f"⚠️ No se encontró el archivo de métricas de Fase 4.1: {RUTA_SALIDA_METRICAS}")
        return

    df_metrics = pd.read_csv(RUTA_SALIDA_METRICAS)
    if df_metrics.empty:
        print("⚠️ El archivo de métricas está vacío.")
        return

    df_cand = df_metrics[df_metrics["es_candidato_subclustering"] == 1].copy()
    if df_cand.empty:
        print("⚠️ No hay clusters marcados como candidatos a subclustering.")
        return

    print("\nClusters candidatos a subclustering:")
    print(df_cand[["sitio", "cluster_hdbscan", "n_segmentos", "dist_media", "dist_p90"]])

    for _, row in df_cand.iterrows():
        sitio = row["sitio"]
        k_cluster = int(row["cluster_hdbscan"])
        print(f"\nProcesando sitio={sitio} | cluster={k_cluster}")
        procesar_sitio_y_cluster(sitio, k_cluster)


if __name__ == "__main__":
    main()