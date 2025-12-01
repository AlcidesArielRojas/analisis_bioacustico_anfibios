# ================================================================
# Fase 2: Clustering y exportación de anotaciones Raven
# Autor: Alcides Rojas
# Correo: alcidesrojasg@gmail.com
# Fecha de creación: 2025-11-10
# Descripción: Carga características extraídas en Fase 1 (.parquet),
#              aplica estandarización, reduce dimensionalidad con UMAP,
#              agrupa segmentos con HDBSCAN,
#              genera visualización de clusters y exporta tablas
#              en formato Raven (.Table.1.selections.txt).
# Dependencias: pandas, numpy, scikit-learn, umap-learn, hdbscan,
#               matplotlib, seaborn
# Asistencia: Microsoft Copilot (IA)
# ================================================================


# fase2_clustering.py
import pandas as pd
# import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuración ---
ruta_salida = Path("resultados")
ruta_features = ruta_salida / "features.parquet"
ruta_raven = ruta_salida / "anotaciones_raven"
ruta_raven.mkdir(exist_ok=True)

UMAP_N_NEIGHBORS = 80
UMAP_MIN_DIST = 0.1
HDBSCAN_MIN_PTS = 100
LIMITE_INFERIOR_HZ = 700
LIMITE_SUPERIOR_HZ = 2500

# --- Orquestador ---
if __name__ == "__main__":
    datos = pd.read_parquet(ruta_features)
    columnas = [c for c in datos.columns if "mfcc" in c]
    escalados = StandardScaler().fit_transform(datos[columnas])

    print("Ejecutando UMAP...")
    reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, random_state=123)
    embedding = reducer.fit_transform(escalados)

    print("Ejecutando HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_PTS)
    clusterer.fit(embedding)
    datos["cluster"] = clusterer.labels_

    # Visualización
    plot_data = pd.DataFrame(embedding, columns=["UMAP1","UMAP2"])
    plot_data["cluster"] = datos["cluster"]
    plt.figure(figsize=(12,9))
    sns.scatterplot(x="UMAP1", y="UMAP2", hue="cluster", data=plot_data, palette="tab20", s=10)
    plt.savefig(ruta_salida/"clusters.png")

    # Exportación Raven
    def exportar_raven(grupo):
        nombre = grupo["archivo_origen"].iloc[0]
        tabla = pd.DataFrame({
            "Selection": range(1,len(grupo)+1),
            "View": "Spectrogram 1",
            "Channel": 1,
            "Begin Time (s)": grupo["tiempo_inicio"],
            "End Time (s)": grupo["tiempo_fin"],
            "Low Freq (Hz)": LIMITE_INFERIOR_HZ,
            "High Freq (Hz)": LIMITE_SUPERIOR_HZ,
            "Cluster_ID": [f"Cluster {c}" if c!=-1 else "Ruido" for c in grupo["cluster"]]
        })
        salida = ruta_raven / f"{Path(nombre).stem}.Table.1.selections.txt"
        tabla.to_csv(salida, sep="\t", index=False)

    datos.groupby("archivo_origen").apply(exportar_raven)
    print("Exportación Raven completa.")