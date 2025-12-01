# ================================================================
# Fase 2: Clustering y exportación de anotaciones Raven (mejorado)
# Autor: Alcides Rojas
# Correo: alcidesrojasg@gmail.com
# Fecha de creación: 2025-11-24
# Descripción: Carga características extraídas en Fase 1 (.parquet),
#              aplica estandarización, reduce dimensionalidad con UMAP,
#              agrupa segmentos con HDBSCAN,
#              exporta tablas Raven enriquecidas con probabilidad,
#              ID global, campo para validación humana y embeddings UMAP.
# ================================================================

import pandas as pd
import hashlib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuración ---
ruta_salida = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local\resultados")
ruta_features = ruta_salida / "features.parquet"
ruta_raven = ruta_salida / "anotaciones_raven"
ruta_raven.mkdir(exist_ok=True)

UMAP_N_NEIGHBORS = 80
UMAP_MIN_DIST = 0.1
HDBSCAN_MIN_PTS = 100
LIMITE_INFERIOR_HZ = 700
LIMITE_SUPERIOR_HZ = 2500

# --- Funciones auxiliares ---
def generar_id_global(row):
    """Genera un hash único por selección usando archivo, tiempos y canal."""
    clave = f"{row['archivo_origen']}|{row['tiempo_inicio']}|{row['tiempo_fin']}|{row.get('channel',1)}"
    return hashlib.md5(clave.encode()).hexdigest()

# --- Orquestador ---
if __name__ == "__main__":
    # Cargar datos de Fase 1
    datos = pd.read_parquet(ruta_features)
    columnas = [c for c in datos.columns if "mfcc" in c]
    escalados = StandardScaler().fit_transform(datos[columnas])

    # UMAP
    print("Ejecutando UMAP...")
    reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, random_state=123)
    embedding = reducer.fit_transform(escalados)
    datos["UMAP1"], datos["UMAP2"] = embedding[:,0], embedding[:,1]

    # HDBSCAN
    print("Ejecutando HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_PTS)
    clusterer.fit(embedding)
    datos["cluster"] = clusterer.labels_
    datos["cluster_prob"] = clusterer.probabilities_

    # ID global y campo para validación humana
    datos["id_global"] = datos.apply(generar_id_global, axis=1)
    datos["species_label"] = ""

    # Guardar resultados
    datos.to_parquet(ruta_salida / "embeddings_clusters.parquet", index=False)

    # Visualización
    plt.figure(figsize=(12,9))
    sns.scatterplot(x="UMAP1", y="UMAP2", hue="cluster", data=datos, palette="tab20", s=10)
    plt.savefig(ruta_salida / "clusters.png")

    # Exportación Raven enriquecida
    def exportar_raven(grupo):
        nombre_rel = grupo["archivo_origen"].iloc[0]  # ej: 'EU-11Tapyta/rec_0001.wav'
        sitio = grupo["sitio"].iloc[0] if "sitio" in grupo.columns else "SITE"
        base = Path(nombre_rel).stem
        tabla = pd.DataFrame({
            "Selection": range(1, len(grupo)+1),
            "View": "Spectrogram 1",
            "Channel": 1,
            "Begin Time (s)": grupo["tiempo_inicio"],
            "End Time (s)": grupo["tiempo_fin"],
            "Low Freq (Hz)": LIMITE_INFERIOR_HZ,
            "High Freq (Hz)": LIMITE_SUPERIOR_HZ,
            "Cluster_ID": [f"Cluster {c}" if c != -1 else "Ruido" for c in grupo["cluster"]],
            "Cluster_Prob": grupo["cluster_prob"],
            "Global_ID": grupo["id_global"],
            "Species_Label": grupo["species_label"],
            "UMAP1": grupo["UMAP1"],
            "UMAP2": grupo["UMAP2"]
        })
        salida = ruta_raven / f"{sitio}_{base}.Table.1.selections.txt"
        tabla.to_csv(salida, sep="\t", index=False)

    datos.groupby("archivo_origen").apply(exportar_raven)
    print("Exportación Raven enriquecida completa.")