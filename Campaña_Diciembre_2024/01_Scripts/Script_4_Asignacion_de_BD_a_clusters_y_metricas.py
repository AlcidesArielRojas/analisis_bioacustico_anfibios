# ============================================================
# Script 4 — Asignación de BD a clusters de campaña + métricas
# ------------------------------------------------------------
# Este script asigna cada segmento de la BD al cluster HDBSCAN
# más cercano en el espacio UMAP 2D de campaña (vecino más
# cercano). Además calcula métricas de calidad por cluster:
#   - distancia media BD→cluster
#   - percentil 90 y 95 de distancia
#   - conteo por especie dentro de cada cluster
# Genera:
#   - CSV detallado por sitio con cluster asignado
#   - CSV resumen por cluster y especie
#   - CSV de métricas de distancia por cluster
# ============================================================

from pathlib import Path
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

NOMBRE_CAMPANIA = "Campaña diciembre 2024"
SUFIJO_CORRIDA = "v2_horario18a06_insectos6a12"

BASE_DIR = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local")
BASE_RESULTADOS = BASE_DIR / "resultados_HDD_Seagate" / NOMBRE_CAMPANIA
RUTA_PROYECCION_BD = BASE_DIR / "proyeccion_BD_fase1_5"


def asignar_clusters_bd_sitio(sitio: str):
    print(f"\n=== Asignación BD → clusters para sitio: {sitio} ===")

    ruta_sitio = BASE_RESULTADOS / sitio
    ruta_fase2 = ruta_sitio / f"{sitio}_{SUFIJO_CORRIDA}_fase2_umap_hdbscan.csv"
    ruta_bd = RUTA_PROYECCION_BD / f"{sitio}_{SUFIJO_CORRIDA}_proyeccion_BD_umap_real.csv"

    if not ruta_fase2.exists():
        print(f"⚠️ No se encontró Fase 2 para {sitio}: {ruta_fase2}")
        return
    if not ruta_bd.exists():
        print(f"⚠️ No se encontró proyección BD para {sitio}: {ruta_bd}")
        return

    df_f2 = pd.read_csv(ruta_fase2)
    df_bd = pd.read_csv(ruta_bd)

    # Verificación de columnas
    if "cluster_hdbscan" not in df_f2.columns:
        print(f"⚠️ No se encontró columna 'cluster_hdbscan' en Fase 2 para {sitio}")
        return

    if not {"BD_DIM1", "BD_DIM2"}.issubset(df_bd.columns):
        print(f"⚠️ No se encontraron columnas BD_DIM1/BD_DIM2 en BD para {sitio}")
        return

    # Coordenadas UMAP 2D
    X_camp = df_f2[["DIM1", "DIM2"]].values
    X_bd = df_bd[["BD_DIM1", "BD_DIM2"]].values

    if X_bd.shape[0] == 0:
        print(f"⚠️ No hay puntos BD para {sitio}")
        return

    # Vecino más cercano
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_camp)
    distancias, indices = nn.kneighbors(X_bd)

    clusters_nn = df_f2.iloc[indices.flatten()]["cluster_hdbscan"].values

    df_bd_asignado = df_bd.copy()
    df_bd_asignado["cluster_nn"] = clusters_nn
    df_bd_asignado["dist_nn_umap2d"] = distancias.flatten()

    # Guardar detalle
    ruta_salida_detalle = RUTA_PROYECCION_BD / f"{sitio}_{SUFIJO_CORRIDA}_BD_con_clusters.csv"
    df_bd_asignado.to_csv(ruta_salida_detalle, index=False)
    print(f"💾 BD con clusters asignados guardada en: {ruta_salida_detalle}")

    # Resumen por cluster y especie
    if "especie_bd" in df_bd_asignado.columns:
        resumen = (
            df_bd_asignado
            .groupby(["cluster_nn", "especie_bd"])
            .size()
            .reset_index(name="n_segmentos")
            .sort_values(["cluster_nn", "n_segmentos"], ascending=[True, False])
        )

        ruta_salida_resumen = RUTA_PROYECCION_BD / f"{sitio}_{SUFIJO_CORRIDA}_resumen_BD_por_cluster_y_especie.csv"
        resumen.to_csv(ruta_salida_resumen, index=False)
        print(f"💾 Resumen BD por cluster y especie guardado en: {ruta_salida_resumen}")

    # Métricas de distancia por cluster
    metricas = (
        df_bd_asignado
        .groupby("cluster_nn")["dist_nn_umap2d"]
        .agg([
            ("n_segmentos_BD", "count"),
            ("dist_media", "mean"),
            ("dist_p90", lambda x: np.percentile(x, 90)),
            ("dist_p95", lambda x: np.percentile(x, 95)),
        ])
        .reset_index()
        .sort_values("cluster_nn")
    )

    ruta_salida_metricas = RUTA_PROYECCION_BD / f"{sitio}_{SUFIJO_CORRIDA}_metricas_distancia_por_cluster.csv"
    metricas.to_csv(ruta_salida_metricas, index=False)
    print(f"💾 Métricas de distancia por cluster guardadas en: {ruta_salida_metricas}")


def main():
    if not BASE_RESULTADOS.exists():
        print(f"⚠️ Carpeta de resultados no existe: {BASE_RESULTADOS}")
        return

    grabadoras = [p.name for p in BASE_RESULTADOS.iterdir() if p.is_dir()]
    grabadoras = sorted(grabadoras)

    print("Sitios detectados para asignación BD → clusters:")
    for g in grabadoras:
        print(" -", g)

    for sitio in grabadoras:
        asignar_clusters_bd_sitio(sitio)

    print("\n✅ Asignación BD → clusters completada.")


if __name__ == "__main__":
    main()
