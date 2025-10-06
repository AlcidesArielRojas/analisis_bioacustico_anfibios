# ===================================================================
# SCRIPT DE ANÁLISIS BIOACÚSTICO NO SUPERVISADO (VERSIÓN KAGGLE)
# Workflow: Preprocesamiento -> Segmentación -> MFCCs -> UMAP -> HDBSCAN
# ===================================================================

# --- Parte 1: Importar Librerías ---
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy import signal
import soundfile as sf  # Para leer .wav y .flac
import librosa
import umap
import hdbscan
import noisereduce as nr # Para la reducción de ruido
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ===================================================================
# --- Parte 2: Configuración Principal (Adaptada para Kaggle) ---
# ===================================================================

# --- Rutas para el entorno de Kaggle ---
ruta_carpeta_audio = Path("/kaggle/input/rfcx-species-audio-detection/train")
ruta_carpeta_salida = Path("/kaggle/working/salida_analisis")

# --- Parámetros de Análisis ---
DURACION_SEGMENTO_SEG = 4
LIMITE_INFERIOR_HZ = 700
LIMITE_SUPERIOR_HZ = 2500
NUM_MFCC = 20
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
HDBSCAN_MIN_PTS = 20
NUM_WORKERS = os.cpu_count() - 2 if os.cpu_count() > 2 else 1

# ===================================================================
# --- Parte 3: Funciones de Procesamiento ---
# ===================================================================

def procesar_segmento(segmento, frecuencia_muestreo):
    """
    Toma un segmento de audio, lo filtra, reduce el ruido y extrae características MFCC.
    """
    try:
        nyquist = 0.5 * frecuencia_muestreo
        b, a = signal.butter(4, [LIMITE_INFERIOR_HZ / nyquist, LIMITE_SUPERIOR_HZ / nyquist], btype='band')
        sonido_filtrado = signal.filtfilt(b, a, segmento).astype(np.float32)

        sonido_reducido = nr.reduce_noise(y=sonido_filtrado, sr=frecuencia_muestreo, stationary=True)

        mfccs = librosa.feature.mfcc(y=sonido_reducido, sr=frecuencia_muestreo, n_mfcc=NUM_MFCC)
        
        return np.hstack([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)])
    except Exception:
        return None

def procesar_archivo(ruta_archivo):
    """
    Lee un archivo de audio (.wav o .flac), lo segmenta y procesa cada segmento.
    """
    try:
        audio_completo, frecuencia_muestreo = sf.read(ruta_archivo)
        
        if audio_completo.ndim > 1:
            audio_completo = audio_completo.mean(axis=1)

        duracion_total = len(audio_completo) / frecuencia_muestreo
        if duracion_total < DURACION_SEGMENTO_SEG:
            return None
            
        resultados_archivo = []
        inicios = np.arange(0, duracion_total - DURACION_SEGMENTO_SEG, DURACION_SEGMENTO_SEG)
        
        for t_inicio in inicios:
            start_sample = int(t_inicio * frecuencia_muestreo)
            end_sample = int((t_inicio + DURACION_SEGMENTO_SEG) * frecuencia_muestreo)
            segmento = audio_completo[start_sample:end_sample]
            
            vector_caracteristicas = procesar_segmento(segmento, frecuencia_muestreo)
            
            if vector_caracteristicas is not None:
                info_segmento = {
                    'archivo_origen': ruta_archivo.name,
                    'tiempo_inicio': t_inicio,
                    'tiempo_fin': t_inicio + DURACION_SEGMENTO_SEG
                }
                nombres_features = [f'mfcc_mean_{i+1}' for i in range(NUM_MFCC)] + \
                                   [f'mfcc_sd_{i+1}' for i in range(NUM_MFCC)]
                
                fila_resultado = {**info_segmento, **dict(zip(nombres_features, vector_caracteristicas))}
                resultados_archivo.append(fila_resultado)
        
        return pd.DataFrame(resultados_archivo)
    except Exception as e:
        print(f"\nERROR al procesar el archivo {ruta_archivo.name}: {e}")
        return None

# ===================================================================
# --- Parte 4: Script Principal - Orquestador del Análisis ---
# ===================================================================

if __name__ == '__main__':
    # --- A. Preparación ---
    ruta_anotaciones_raven = ruta_carpeta_salida / "anotaciones_raven"
    ruta_carpeta_salida.mkdir(parents=True, exist_ok=True)
    ruta_anotaciones_raven.mkdir(exist_ok=True)
    
    print(f"Configurado para usar {NUM_WORKERS} núcleos de CPU.")
    
    # Localizar todos los archivos de audio, tanto .wav como .flac
    lista_archivos_wav = list(ruta_carpeta_audio.rglob("*.wav"))
    lista_archivos_flac = list(ruta_carpeta_audio.rglob("*.flac"))
    lista_archivos = lista_archivos_wav + lista_archivos_flac
    print(f"Se encontraron {len(lista_archivos)} archivos de audio para procesar.")

    lista_global_resultados = []

    # --- B. Procesamiento Paralelo ---
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futuros = {executor.submit(procesar_archivo, ruta): ruta for ruta in lista_archivos}
        
        for futuro in tqdm(as_completed(futuros), total=len(lista_archivos), desc="Procesando archivos"):
            resultado_df = futuro.result()
            if resultado_df is not None and not resultado_df.empty:
                lista_global_resultados.append(resultado_df)

    # --- C. Consolidación Final de Datos ---
    print("\nConsolidando todos los resultados...")
    if lista_global_resultados:
        datos_completos = pd.concat(lista_global_resultados, ignore_index=True)
        print(f"¡Procesamiento completado! Se extrajeron características de {len(datos_completos)} segmentos de audio.")
    else:
        datos_completos = pd.DataFrame()
        print("¡Procesamiento completado! No se extrajeron características de ningún segmento.")

    if not datos_completos.empty:
        # ===================================================================
        # --- Parte 5: Clustering y Visualización ---
        # ===================================================================

        # --- A. Preparación de Datos para Clustering ---
        columnas_mfcc = [col for col in datos_completos.columns if 'mfcc' in col]
        datos_escalados = StandardScaler().fit_transform(datos_completos[columnas_mfcc])

        # --- B. UMAP ---
        print("Ejecutando UMAP...")
        reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, random_state=123)
        embedding = reducer.fit_transform(datos_escalados)

        # --- C. HDBSCAN ---
        print("Ejecutando HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_PTS)
        clusterer.fit(embedding)
        datos_completos['cluster'] = clusterer.labels_

        # --- D. Visualización ---
        print("Generando gráfico de visualización...")
        plot_data = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
        plot_data['cluster_label'] = [f"Cluster {l}" if l != -1 else "Ruido" for l in clusterer.labels_]
        
        plt.figure(figsize=(12, 9))
        sns.scatterplot(
            x='UMAP1', y='UMAP2', 
            hue='cluster_label', 
            data=plot_data,
            palette="viridis",
            alpha=0.6, s=10
        )
        
        num_clusters_found = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
        num_noise_points = np.sum(clusterer.labels_ == -1)
        plt.title(f"Clusters de Sonidos ({num_clusters_found} grupos encontrados y {num_noise_points} puntos de ruido)")
        plt.xlabel("Dimensión UMAP 1")
        plt.ylabel("Dimensión UMAP 2")
        plt.legend(title="Grupo")
        
        plt.savefig(ruta_carpeta_salida / "visualizacion_clusters.png")
        # En un notebook de Kaggle, plt.show() puede ser redundante, pero no hace daño
        plt.show()

        # ===================================================================
        # --- Parte 6: Generación de Anotaciones para Raven ---
        # ===================================================================
        print("Generando archivos de anotación para Raven...")

        def escribir_anotaciones_raven(data_grupo):
            nombre_archivo = data_grupo['archivo_origen'].iloc[0]
            raven_table = pd.DataFrame({
                'Selection': range(1, len(data_grupo) + 1),
                'View': 'Spectrogram 1',
                'Channel': 1,
                'Begin Time (s)': data_grupo['tiempo_inicio'],
                'End Time (s)': data_grupo['tiempo_fin'],
                'Low Freq (Hz)': LIMITE_INFERIOR_HZ,
                'High Freq (Hz)': LIMITE_SUPERIOR_HZ,
                'Cluster_ID': [f"Cluster {c}" if c != -1 else "Ruido" for c in data_grupo['cluster']]
            })
            nombre_base = Path(nombre_archivo).stem
            ruta_salida_txt = ruta_anotaciones_raven / f"{nombre_base}.Table.1.selections.txt"
            raven_table.to_csv(ruta_salida_txt, sep='\t', index=False)
            return None

        # Agrupar por archivo y aplicar la función
        datos_completos.groupby('archivo_origen').apply(escribir_anotaciones_raven)
        
        print(f"\n¡PROCESO COMPLETADO! Los resultados se encuentran en: {ruta_carpeta_salida}")