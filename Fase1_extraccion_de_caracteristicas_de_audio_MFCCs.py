# ================================================================
# Fase 1: Extracción de características de audio (MFCCs) optimizada
# Autor: Alcides Rojas
# Fecha: 2025-12-10
# Modificaciones: validación de archivos, segmentación manual,
#                 control de errores, compatibilidad con notebook
# ================================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
from scipy import signal
import librosa
import noisereduce as nr
from tqdm import tqdm
from joblib import Parallel, delayed

# --- Configuración ---
DURACION_SEGMENTO_SEG = 4
SOLAPAMIENTO_SEG = 2
LIMITE_INFERIOR_HZ = 700
LIMITE_SUPERIOR_HZ = 2500
NUM_MFCC = 20
NUM_WORKERS = 2
TAMANO_BLOQUE = 200

# Disco externo (ajustado para notebook)
ruta_base_externa = Path(r"D:\\")  # Verificá que sea la letra correcta
ruta_salida = Path(r"C:\Users\Alcides\Proyecto_Paisajes_Sonoros_Repositorio_Local\resultados")
ruta_salida.mkdir(exist_ok=True)
ruta_features = ruta_salida / "features.parquet"

# --- Funciones ---
def procesar_segmento(segmento, sr):
    """Aplica filtrado, reducción de ruido y calcula MFCCs de un segmento."""
    try:
        nyquist = 0.5 * sr
        b, a = signal.butter(4, [LIMITE_INFERIOR_HZ/nyquist, LIMITE_SUPERIOR_HZ/nyquist], btype='band')
        filtrado = signal.filtfilt(b, a, segmento).astype(np.float32)
        reducido = filtrado  # noisereduce desactivado temporalmente
        mfccs = librosa.feature.mfcc(y=reducido, sr=sr, n_mfcc=NUM_MFCC)
        return np.hstack([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)])
    except Exception as e:
        print(f"Error en procesar_segmento: {e}")
        return None

def procesar_archivo(ruta: Path):
    """Procesa un archivo de audio en segmentos solapados de 4s."""
    try:
        if not ruta.exists() or ruta.stat().st_size == 0:
            print(f"Archivo vacío o inexistente: {ruta.name}")
            return None

        audio, sr = sf.read(ruta)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        duracion = len(audio) / sr
        if duracion < DURACION_SEGMENTO_SEG:
            print(f"Audio demasiado corto (<{DURACION_SEGMENTO_SEG}s): {ruta.name}")
            return None

        frame_length = int(DURACION_SEGMENTO_SEG * sr)
        hop_length = int(SOLAPAMIENTO_SEG * sr)
        resultados = []

        for inicio in range(0, len(audio) - frame_length + 1, hop_length):
            seg = audio[inicio:inicio + frame_length]
            vec = procesar_segmento(seg, sr)
            if vec is not None:
                partes = ruta.parts
                if "Data" in partes:
                    idx = partes.index("Data")
                    sitio = partes[idx - 1] if idx > 0 else ruta.parent.name
                else:
                    sitio = ruta.parent.name
                archivo_rel = f"{sitio}/{ruta.name}"

                info = {
                    "archivo_origen": archivo_rel,
                    "sitio": sitio,
                    "tiempo_inicio": round(inicio / sr, 2),
                    "tiempo_fin": round((inicio + frame_length) / sr, 2)
                }
                nombres = [f"mfcc_mean_{j+1}" for j in range(NUM_MFCC)] + \
                          [f"mfcc_sd_{j+1}" for j in range(NUM_MFCC)]
                resultados.append({**info, **dict(zip(nombres, vec))})

        return pd.DataFrame(resultados) if resultados else None
    except Exception as e:
        print(f"Error procesando {ruta.name}: {e}")
        return None

# --- Orquestador ---
if __name__ == "__main__":
    carpetas_site = [ruta_base_externa / "BO-81Tapyta", ruta_base_externa / "BO-82Tapyta"]
    resultados = []

    for carpeta in carpetas_site:
        data_dir = carpeta / 'Data'
        if data_dir.exists():
            archivos = list(data_dir.rglob('*.wav'))
            print(f"Procesando sitio: {carpeta.name} ({len(archivos)} archivos)")
            bloques = np.array_split(np.array(archivos), max(1, len(archivos) // TAMANO_BLOQUE))

            for i, bloque in enumerate(bloques):
                print(f"  → Lote {i+1}/{len(bloques)} con {len(bloque)} archivos")
                resultados_bloque = Parallel(n_jobs=NUM_WORKERS)(
                    delayed(procesar_archivo)(r) for r in bloque
                )
                resultados.extend([df for df in resultados_bloque if df is not None])

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
