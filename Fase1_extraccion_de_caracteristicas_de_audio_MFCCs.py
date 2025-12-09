# ================================================================
# Fase 1: Extracción de características de audio (MFCCs)
# Autor: Alcides Rojas
# Correo: alcidesrojasg@gmail.com
# Fecha de creación: 2025-11-10
# Descripción: Procesa audios .wav/.flac en subcarpetas 'Data' del disco externo,
#              segmenta en ventanas de 4s, aplica filtro pasa banda y reducción de ruido,
#              calcula MFCCs y guarda resultados en formato .parquet.
# Dependencias: numpy, pandas, librosa, soundfile, scipy, noisereduce, tqdm
# Asistencia: Microsoft Copilot (IA)
# ================================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import soundfile as sf
from scipy import signal
import librosa
import noisereduce as nr
from tqdm import tqdm

# --- Configuración ---
DURACION_SEGMENTO_SEG = 4
LIMITE_INFERIOR_HZ = 700
LIMITE_SUPERIOR_HZ = 2500
NUM_MFCC = 20
NUM_WORKERS = 4
TAMANO_BLOQUE = 1000  # cantidad de archivos por lote

# Disco externo (ajusta la letra si no es D:)
ruta_base_externa = Path(r"D:\\")  # SAMSUNG (D:)
# Carpeta local del repo, donde guardarás resultados
ruta_salida = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local\resultados")
ruta_salida.mkdir(exist_ok=True)

ruta_features = ruta_salida / "features.parquet"

# --- Funciones ---
def procesar_segmento(segmento, sr):
    try:
        nyquist = 0.5 * sr
        b, a = signal.butter(4, [LIMITE_INFERIOR_HZ/nyquist, LIMITE_SUPERIOR_HZ/nyquist], btype='band')
        filtrado = signal.filtfilt(b, a, segmento).astype(np.float32)
        reducido = nr.reduce_noise(y=filtrado, sr=sr, stationary=False)
        mfccs = librosa.feature.mfcc(y=reducido, sr=sr, n_mfcc=NUM_MFCC)
        return np.hstack([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)])
    except Exception as e:
        print(f"Error en procesar_segmento: {e}")
        return None

def procesar_archivo(ruta: Path):
    try:
        audio, sr = sf.read(ruta)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        duracion = len(audio) / sr
        if duracion < DURACION_SEGMENTO_SEG:
            return None

        resultados = []
        inicios = np.arange(0, duracion - DURACION_SEGMENTO_SEG, DURACION_SEGMENTO_SEG)
        for t in inicios:
            seg = audio[int(t*sr):int((t + DURACION_SEGMENTO_SEG)*sr)]
            vec = procesar_segmento(seg, sr)
            if vec is not None:
                sitio = ruta.parent.parent.name if ruta.parent.name == "Data" else ruta.parent.name
                archivo_rel = f"{sitio}/{ruta.name}"

                info = {
                    "archivo_origen": archivo_rel,
                    "sitio": sitio,
                    "tiempo_inicio": t,
                    "tiempo_fin": t + DURACION_SEGMENTO_SEG
                }
                nombres = [f"mfcc_mean_{i+1}" for i in range(NUM_MFCC)] + \
                          [f"mfcc_sd_{i+1}" for i in range(NUM_MFCC)]
                resultados.append({**info, **dict(zip(nombres, vec))})

        return pd.DataFrame(resultados) if resultados else None
    except Exception as e:
        print(f"Error en {ruta.name}: {e}")
        return None

# --- Orquestador ---
if __name__ == "__main__":
    carpetas_site = [p for p in ruta_base_externa.iterdir() if p.is_dir()]
    resultados = []

    for carpeta in carpetas_site:
        data_dir = carpeta / 'Data'
        if data_dir.exists():
            archivos = list(data_dir.rglob('*.wav'))
            # También hay .flac en Data, podés incluir:
            # archivos += list(data_dir.rglob('*.flac'))

            print(f"Procesando sitio: {carpeta.name} ({len(archivos)} archivos)")
            bloques = np.array_split(archivos, max(1, len(archivos) // TAMANO_BLOQUE))

            for i, bloque in enumerate(bloques):
                print(f"  → Lote {i+1}/{len(bloques)} con {len(bloque)} archivos")
                with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
                    futuros = {ex.submit(procesar_archivo, r): r for r in bloque}
                    for f in tqdm(as_completed(futuros), total=len(bloque), desc=f"{carpeta.name} - Lote {i+1}"):
                        try:
                            df = f.result()
                            if df is not None and not df.empty:
                                resultados.append(df)
                        except Exception as e:
                            print(f"⚠️ Error en procesamiento paralelo: {e}")

    if resultados:
        datos = pd.concat(resultados, ignore_index=True)
        datos.to_parquet(ruta_features, index=False)
        print(f"Guardado en {ruta_features}, tiene {len(datos)} segmentos procesados.")

        resumen = datos.groupby("sitio").size().reset_index(name="segmentos")
        resumen.to_csv(ruta_salida / "resumen_segmentos_por_sitio.csv", index=False)
        print("Resumen por sitio:")
        print(resumen)
    else:
        print("No se generaron resultados. Verifica que existan .wav en las carpetas Data del disco externo.")
