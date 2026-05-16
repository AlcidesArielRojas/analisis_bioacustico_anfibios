# ================================================================
# Fase 4.X: Contexto de clusters en el UMAP original (resumen)
# ------------------------------------------------
# DESCRIPCIÓN BREVE
#
# Este script:
#   1) Elimina las figuras viejas de contexto
#      (<sitio>_cluster*_contexto_umap2d.png).
#   2) Lee:
#        - Fase 4.1: clusters candidatos a subclustering,
#        - Fase 2: UMAP + cluster_hdbscan por sitio,
#        - Fase 4.2: subclusters por sitio.
#   3) Para cada sitio genera un solo UMAP 2D donde:
#        - todos los clusters originales se ven en gris,
#        - los clusters candidatos se colorean (p.ej. naranja),
#        - los clusters que efectivamente tuvieron subclustering
#          se colorean distinto (p.ej. rojo),
#        - se anota el número de cluster sobre los que sí se
#          subclusterizaron.
#
# No modifica datos, solo borra figuras viejas de contexto y crea
# nuevas figuras de resumen por sitio.
# ================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- CONFIGURACIÓN ----------------

NOMBRE_CAMPANIA = "Campaña diciembre 2024"
SUFIJO_CORRIDA = "v2_horario18a06_insectos6a12"

BASE_DIR = Path(r"C:/Users/User/Proyecto_Paisajes_Sonoros_Repositorio_Local")
BASE_RESULTADOS = BASE_DIR / "resultados_HDD_Seagate" / NOMBRE_CAMPANIA

# Archivo de métricas de Fase 4.1 (donde están los candidatos)
RUTA_SALIDA_METRICAS = BASE_RESULTADOS / f"{NOMBRE_CAMPANIA}_{SUFIJO_CORRIDA}_fase4_clusters_metrics.csv"

# Carpeta donde se guardan las figuras de subclusters (Fase 4.3)
RUTA_FIGURAS = BASE_DIR / "figuras_inspeccion_subclusters" / NOMBRE_CAMPANIA
RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["savefig.dpi"] = 150


# ---------------- UTILIDADES ----------------

def borrar_figuras_contexto_viejas():
    """
    Elimina todos los archivos *_contexto_umap2d.png
    dentro de la carpeta de figuras de la campaña.
    """
    print("\nEliminando figuras viejas de contexto...")
    n_borrados = 0
    for sitio_dir in RUTA_FIGURAS.iterdir():
        if not sitio_dir.is_dir():
            continue
        for png in sitio_dir.glob("*_contexto_umap2d.png"):
            png.unlink()
            n_borrados += 1
    print(f"✓ Figuras de contexto eliminadas: {n_borrados}")


def cargar_candidatos():
    """
    Lee el archivo de métricas de Fase 4.1 y devuelve
    un DataFrame con los clusters marcados como candidatos.
    """
    if not RUTA_SALIDA_METRICAS.exists():
        print(f"⚠️ No se encontró el archivo de métricas de Fase 4.1: {RUTA_SALIDA_METRICAS}")
        return None

    df_metrics = pd.read_csv(RUTA_SALIDA_METRICAS)
    if df_metrics.empty:
        print("⚠️ El archivo de métricas está vacío.")
        return None

    if "es_candidato_subclustering" not in df_metrics.columns:
        print("⚠️ No se encontró la columna 'es_candidato_subclustering' en el archivo de métricas.")
        return None

    df_cand = df_metrics[df_metrics["es_candidato_subclustering"] == 1].copy()
    if df_cand.empty:
        print("⚠️ No hay clusters marcados como candidatos a subclustering.")
        return None

    return df_cand


def obtener_clusters_con_subclustering(sitio: str):
    """
    A partir del CSV de Fase 4.2 para un sitio, devuelve el conjunto
    de cluster_hdbscan que efectivamente tienen subclusters válidos
    (subcluster_id >= 0).
    """
    ruta_csv_fase4_2 = BASE_RESULTADOS / sitio / f"{sitio}_{SUFIJO_CORRIDA}_fase4_umap_hdbscan_subclusters.csv"
    if not ruta_csv_fase4_2.exists():
        return set()

    df4 = pd.read_csv(ruta_csv_fase4_2)
    if df4.empty:
        return set()

    df_valid = df4[(df4["cluster_hdbscan"] != -1) & (df4["subcluster_id"] >= 0)]
    if df_valid.empty:
        return set()

    return set(df_valid["cluster_hdbscan"].unique())


def graficar_resumen_contexto_sitio(sitio: str, df_cand_sitio: pd.DataFrame):
    """
    Para un sitio:
      - carga Fase 2 (UMAP + cluster_hdbscan),
      - identifica clusters candidatos (Fase 4.1),
      - identifica clusters con subclustering real (Fase 4.2),
      - genera un UMAP 2D con:
          * todos los puntos en gris,
          * clusters candidatos en naranja,
          * clusters con subclustering en rojo,
          * texto con el número de cluster sobre los que se subclusterizaron.
    """
    ruta_csv_fase2 = BASE_RESULTADOS / sitio / f"{sitio}_{SUFIJO_CORRIDA}_fase2_umap_hdbscan.csv"
    if not ruta_csv_fase2.exists():
        print(f"⚠️ No se encontró CSV de Fase 2 para {sitio}: {ruta_csv_fase2}")
        return

    df2 = pd.read_csv(ruta_csv_fase2)
    for col in ["U1", "U2", "cluster_hdbscan"]:
        if col not in df2.columns:
            print(f"⚠️ {sitio}: falta columna {col} en {ruta_csv_fase2}")
            return

    # Conjuntos de clusters
    clusters_candidatos = set(df_cand_sitio["cluster_hdbscan"].unique())
    clusters_subclustering = obtener_clusters_con_subclustering(sitio)

    U1_all = df2["U1"].values
    U2_all = df2["U2"].values
    cl_all = df2["cluster_hdbscan"].values

    fig, ax = plt.subplots(figsize=(8, 6))

    # 1) Todos los puntos en gris claro
    ax.scatter(U1_all, U2_all, c="lightgray", s=5, alpha=0.4, label="Otros clusters")

    # 2) Clusters candidatos (naranja), aunque no hayan tenido subclustering
    mask_cand = np.isin(cl_all, list(clusters_candidatos))
    if mask_cand.any():
        ax.scatter(
            U1_all[mask_cand],
            U2_all[mask_cand],
            c="orange",
            s=8,
            alpha=0.8,
            label="Clusters candidatos"
        )

    # 3) Clusters con subclustering real (rojo)
    mask_sub = np.isin(cl_all, list(clusters_subclustering))
    if mask_sub.any():
        ax.scatter(
            U1_all[mask_sub],
            U2_all[mask_sub],
            c="red",
            s=10,
            alpha=0.9,
            label="Clusters con subclustering"
        )

        # Anotar número de cluster sobre su centro aproximado
        for k in sorted(clusters_subclustering):
            mask_k = cl_all == k
            if not mask_k.any():
                continue
            x_mean = U1_all[mask_k].mean()
            y_mean = U2_all[mask_k].mean()
            ax.text(
                x_mean,
                y_mean,
                str(int(k)),
                color="black",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )

    ax.set_title(f"{sitio} | UMAP 2D contexto de clusters", fontsize=11)
    ax.set_xlabel("U1")
    ax.set_ylabel("U2")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()

    carpeta_figuras_sitio = RUTA_FIGURAS / sitio
    carpeta_figuras_sitio.mkdir(parents=True, exist_ok=True)

    nombre_fig = f"{sitio}_contexto_clusters_resumen_umap2d.png"
    ruta_fig = carpeta_figuras_sitio / nombre_fig
    fig.savefig(ruta_fig)
    plt.close(fig)

    print(f"✓ Figura de contexto RESUMEN guardada: {ruta_fig}")


def main():
    # 1) Borrar figuras viejas de contexto
    borrar_figuras_contexto_viejas()

    # 2) Cargar clusters candidatos (Fase 4.1)
    df_cand = cargar_candidatos()
    if df_cand is None:
        return

    print("\nClusters candidatos a visualizar en contexto UMAP (resumen):")
    print(df_cand[["sitio", "cluster_hdbscan", "n_segmentos"]])

    # 3) Procesar sitio por sitio
    for sitio, df_cand_sitio in df_cand.groupby("sitio"):
        print(f"\nProcesando contexto RESUMEN | sitio={sitio}")
        graficar_resumen_contexto_sitio(sitio, df_cand_sitio)


if __name__ == "__main__":
    main()