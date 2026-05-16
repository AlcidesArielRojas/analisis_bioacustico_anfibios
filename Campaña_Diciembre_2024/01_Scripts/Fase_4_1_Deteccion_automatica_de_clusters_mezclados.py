# ================================================================
# Fase 4.1: Detección automática de clusters "mezclados" o dispersos
# ------------------------------------------------
# Qué hace este script (en sencillo):
# - Recorre todos los sitios (grabadoras) de una campaña en resultados_HDD_Seagate.
# - Para cada sitio, abre el CSV de Fase 2 (UMAP + HDBSCAN).
# - Para cada cluster_hdbscan (≠ -1):
#     * cuenta cuántos segmentos tiene,
#     * calcula la distancia media al centroide en UMAP (U1, U2, U3),
#     * calcula el percentil 90 de esa distancia.
# - Marca como "candidato a subclustering" a los clusters grandes y dispersos.
# - Guarda un CSV con estas métricas por sitio y cluster.
# ================================================================

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------- CONFIGURACIÓN (MODIFICABLE) ----------------

# Nombre de la campaña (debe coincidir con Fase 1 y Fase 2)
NOMBRE_CAMPANIA = "Campaña diciembre 2024"

# Sufijo de la corrida (debe coincidir con Fase 2)
SUFIJO_CORRIDA = "v2_horario18a06_insectos6a12"

# Ruta base de tu proyecto
BASE_DIR = Path(r"C:/Users/User/Proyecto_Paisajes_Sonoros_Repositorio_Local")

# Carpeta de resultados de esta campaña (igual que en Fase 2)
BASE_RESULTADOS = BASE_DIR / "resultados_HDD_Seagate" / NOMBRE_CAMPANIA

# Umbrales para marcar clusters como "candidatos a subclustering"
MIN_SEGMENTOS_CANDIDATO = 1000      # mínimo de segmentos en el cluster
UMBRAL_DIST_MEDIA = 0.8             # umbral de distancia media al centroide (ajustable)
UMBRAL_DIST_P90 = 1.2               # umbral de percentil 90 de distancia (ajustable)

# Archivo de salida con métricas por cluster
RUTA_SALIDA_METRICAS = BASE_RESULTADOS / f"{NOMBRE_CAMPANIA}_{SUFIJO_CORRIDA}_fase4_clusters_metrics.csv"


def analizar_clusters_sitio(sitio: str) -> pd.DataFrame | None:
    """
    Lee el CSV de Fase 2 para un sitio y calcula métricas por cluster_hdbscan.
    Devuelve un DataFrame con una fila por cluster.
    """
    ruta_csv = BASE_RESULTADOS / sitio / f"{sitio}_{SUFIJO_CORRIDA}_fase2_umap_hdbscan.csv"
    if not ruta_csv.exists():
        print(f"⚠️ No se encontró CSV de Fase 2 para {sitio}: {ruta_csv}")
        return None

    df = pd.read_csv(ruta_csv)

    # Verificar columnas necesarias
    for col in ["U1", "U2", "U3", "cluster_hdbscan"]:
        if col not in df.columns:
            print(f"⚠️ {sitio}: falta columna {col} en {ruta_csv}")
            return None

    df_valid = df[df["cluster_hdbscan"] != -1].copy()
    if df_valid.empty:
        print(f"⚠️ {sitio}: no hay clusters válidos (todos ruido).")
        return None

    filas = []
    for k, df_k in df_valid.groupby("cluster_hdbscan"):
        coords = df_k[["U1", "U2", "U3"]].values
        if len(coords) < 2:
            continue

        centroide = coords.mean(axis=0)
        dist = np.linalg.norm(coords - centroide, axis=1)

        n_seg = len(df_k)
        dist_media = float(dist.mean())
        dist_p90 = float(np.percentile(dist, 90))

        candidato = (
            (n_seg >= MIN_SEGMENTOS_CANDIDATO)
            and ((dist_media >= UMBRAL_DIST_MEDIA) or (dist_p90 >= UMBRAL_DIST_P90))
        )

        filas.append({
            "sitio": sitio,
            "cluster_hdbscan": int(k),
            "n_segmentos": int(n_seg),
            "dist_media": dist_media,
            "dist_p90": dist_p90,
            "es_candidato_subclustering": int(candidato),
        })

    if not filas:
        return None

    return pd.DataFrame(filas)


def main():
    if not BASE_RESULTADOS.exists():
        print(f"⚠️ No existe la carpeta de resultados: {BASE_RESULTADOS}")
        return

    sitios = [p.name for p in BASE_RESULTADOS.iterdir() if p.is_dir()]
    sitios = sorted(sitios)

    print("\nSitios detectados para Fase 4.1 (detección de clusters mezclados):")
    for s in sitios:
        print(" -", s)

    dfs_all = []
    for sitio in sitios:
        print(f"\nAnalizando sitio: {sitio}")
        df_metrics = analizar_clusters_sitio(sitio)
        if df_metrics is not None:
            dfs_all.append(df_metrics)

    if not dfs_all:
        print("⚠️ No se generaron métricas para ningún sitio.")
        return

    df_total = pd.concat(dfs_all, ignore_index=True)
    df_total.to_csv(RUTA_SALIDA_METRICAS, index=False)
    print(f"\n✅ Métricas de clusters guardadas en: {RUTA_SALIDA_METRICAS}")

    print("\nResumen de clusters candidatos a subclustering:")
    df_cand = df_total[df_total["es_candidato_subclustering"] == 1]
    if df_cand.empty:
        print(" - Ninguno según los umbrales actuales.")
    else:
        print(df_cand[["sitio", "cluster_hdbscan", "n_segmentos", "dist_media", "dist_p90"]])


if __name__ == "__main__":
    main()