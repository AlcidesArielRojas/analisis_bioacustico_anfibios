# ================================================================
# Fase 1: Extracción de características de audio (MFCCs)
# Autor: Alcides Rojas
# Correo: alcidesrojasg@gmail.com
# Fecha de creación: 2025-11-10
# Descripción: Procesa audios .wav/.flac, segmenta en ventanas de 4s,
#              aplica filtro pasa banda y reducción de ruido,
#              calcula MFCCs y guarda resultados en formato .parquet.
# Dependencias: numpy, pandas, librosa, soundfile, umap-learn, hdbscan, noisereduce
# Asistencia: Microsoft Copilot (IA)
# ================================================================


# fase1_features.py
import os
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
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
NUM_WORKERS = os.cpu_count() - 2 if os.cpu_count() > 2 else 1

ruta_carpeta_audio = Path("input_wav")
ruta_salida = Path("resultados")
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
    except Exception:
        return None

def procesar_archivo(ruta):
    try:
        audio, sr = sf.read(ruta)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        duracion = len(audio)/sr
        if duracion < DURACION_SEGMENTO_SEG:
            return None
        resultados = []
        inicios = np.arange(0, duracion-DURACION_SEGMENTO_SEG, DURACION_SEGMENTO_SEG)
        for t in inicios:
            seg = audio[int(t*sr):int((t+DURACION_SEGMENTO_SEG)*sr)]
            vec = procesar_segmento(seg, sr)
            if vec is not None:
                info = {
                    "archivo_origen": ruta.name,
                    "tiempo_inicio": t,
                    "tiempo_fin": t+DURACION_SEGMENTO_SEG
                }
                nombres = [f"mfcc_mean_{i+1}" for i in range(NUM_MFCC)] + \
                          [f"mfcc_sd_{i+1}" for i in range(NUM_MFCC)]
                resultados.append({**info, **dict(zip(nombres, vec))})
        return pd.DataFrame(resultados)
    except Exception as e:
        print(f"Error en {ruta.name}: {e}")
        return None

# --- Orquestador ---
if __name__ == "__main__":
    archivos = list(ruta_carpeta_audio.rglob("*.wav")) + list(ruta_carpeta_audio.rglob("*.flac"))
    resultados = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futuros = {ex.submit(procesar_archivo, r): r for r in archivos}
        for f in tqdm(as_completed(futuros), total=len(archivos)):
            df = f.result()
            if df is not None and not df.empty:
                resultados.append(df)
    if resultados:
        datos = pd.concat(resultados, ignore_index=True)
        datos.to_parquet(ruta_features)
        print(f"Guardado en {ruta_features}, {len(datos)} segmentos procesados.")