# ================================================================
# Fase 1: Extracción de características de audio (MFCCs) optimizada
# Autor: Alcides Rojas
# Fecha: 2025-12-12
# Modificaciones:
#  - Unificación de sitios BO-81Tapyta y BO-82Tapyta → "BO"
#  - Procesamiento secuencial (sin joblib/Parallel)
#  - Guardado por lote en .parquet y detección de lotes ya procesados
#  - Concatenación final desde temporales
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

# --- Configuración ---
DURACION_SEGMENTO_SEG = 4
SOLAPAMIENTO_SEG = 2   # 50% de solapamiento → hop = 2s
LIMITE_INFERIOR_HZ = 700
LIMITE_SUPERIOR_HZ = 2500
NUM_MFCC = 20
TAMANO_BLOQUE = 100  # cantidad de archivos por lote

# Disco externo (ajusta la letra si no es D:)
ruta_base_externa = Path(r"D:\\")  # SAMSUNG (D:)
# Carpeta local del repo, donde guardarás resultados
ruta_salida = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local\resultados")
ruta_salida.mkdir(exist_ok=True)

ruta_temporales = ruta_salida / "temporales_fase1"
ruta_temporales.mkdir(exist_ok=True)
ruta_features_final = ruta_salida / "features.parquet"
ruta_resumen = ruta_salida / "resumen_segmentos_por_sitio.csv"

# --- Funciones ---
def procesar_segmento(segmento, sr):
    """Aplica filtrado, reducción de ruido y calcula MFCCs de un segmento."""
    try:
        nyquist = 0.5 * sr
        b, a = signal.butter(4, [LIMITE_INFERIOR_HZ/nyquist, LIMITE_SUPERIOR_HZ/nyquist], btype='band')
        filtrado = signal.filtfilt(b, a, segmento).astype(np.float32)
        reducido = nr.reduce_noise(y=filtrado, sr=sr, stationary=False)
        mfccs = librosa.feature.mfcc(y=reducido, sr=sr, n_mfcc=NUM_MFCC)
        return np.hstack([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)])
    except Exception as e:
        print(f"[ERROR] procesar_segmento: {e}")
        return None

def procesar_archivo(ruta: Path):
    """Procesa un archivo de audio en segmentos solapados de 4s."""
    try:
        audio, sr = sf.read(ruta)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        duracion = len(audio) / sr
        if duracion < DURACION_SEGMENTO_SEG:
            return None

        # Ventanas de 4s con hop de 2s
        frame_length = int(DURACION_SEGMENTO_SEG * sr)
        hop_length = int(SOLAPAMIENTO_SEG * sr)
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length).T

        resultados = []
        # Determinación del sitio a partir de la ruta
        partes = ruta.parts
        if "Data" in partes:
            idx = partes.index("Data")
            sitio = partes[idx - 1] if idx > 0 else ruta.parent.name
        else:
            sitio = ruta.parent.name

        # --- Unificación de sitios BO ---
        if sitio.startswith("BO-"):
            sitio = "BO"

        for i, seg in enumerate(frames):
            vec = procesar_segmento(seg, sr)
            if vec is not None:
                archivo_rel = f"{sitio}/{ruta.name}"
                info = {
                    "archivo_origen": archivo_rel,
                    "sitio": sitio,
                    "tiempo_inicio": i * SOLAPAMIENTO_SEG,
                    "tiempo_fin": i * SOLAPAMIENTO_SEG + DURACION_SEGMENTO_SEG
                }
                nombres = [f"mfcc_mean_{j+1}" for j in range(NUM_MFCC)] + \
                          [f"mfcc_sd_{j+1}" for j in range(NUM_MFCC)]
                resultados.append({**info, **dict(zip(nombres, vec))})

        return pd.DataFrame(resultados) if resultados else None
    except Exception as e:
        print(f"[ERROR] {ruta.name}: {e}")
        return None

# --- Orquestador ---
if __name__ == "__main__":
    carpetas_site = [p for p in ruta_base_externa.iterdir() if p.is_dir()]

    for carpeta in carpetas_site:
        data_dir = carpeta / 'Data'
        if not data_dir.exists():
            continue

        archivos = list(data_dir.rglob('*.wav'))
        # Si querés incluir .flac, descomentá:
        # archivos += list(data_dir.rglob('*.flac'))

        print(f"Procesando sitio: {carpeta.name} ({len(archivos)} archivos)")
        if len(archivos) == 0:
            continue

        # Bloques por tamaño
        num_bloques = max(1, len(archivos) // TAMANO_BLOQUE)
        bloques = np.array_split(archivos, num_bloques)

        for i, bloque in enumerate(bloques):
            # Armar nombre del lote por carpeta y índice
            nombre_lote = f"lote_{carpeta.name}_{i+1:03d}.parquet"
            ruta_lote = ruta_temporales / nombre_lote

            if ruta_lote.exists():
                print(f"⏩ Lote {i+1}/{len(bloques)} ya procesado, se salta.")
                continue

            print(f"📦 Lote {i+1}/{len(bloques)} con {len(bloque)} archivos")
            resultados_bloque = []
            for r in bloque:
                df = procesar_archivo(r)
                if df is not None:
                    resultados_bloque.append(df)

            if resultados_bloque:
                df_lote = pd.concat(resultados_bloque, ignore_index=True)
                df_lote.to_parquet(ruta_lote, index=False)
                print(f"✅ Lote guardado: {nombre_lote}")
            else:
                print(f"⚠️ Lote {i+1} sin resultados")

    # Concatenación final desde temporales
    lotes = sorted(ruta_temporales.glob("lote_*.parquet"))
    if len(lotes) == 0:
        print("⚠️ No se encontraron lotes procesados para concatenar.")
    else:
        print(f"🔗 Concatenando {len(lotes)} lotes...")
        dfs = []
        for p in lotes:
            try:
                dfs.append(pd.read_parquet(p))
            except Exception as e:
                print(f"[ERROR] al leer {p.name}: {e}")

        if dfs:
            datos = pd.concat(dfs, ignore_index=True)
            datos.to_parquet(ruta_features_final, index=False)
            print(f"💾 Guardado final en {ruta_features_final} con {len(datos)} segmentos.")

            resumen = datos.groupby("sitio").size().reset_index(name="segmentos")
            resumen.to_csv(ruta_resumen, index=False)
            print("📊 Resumen por sitio:")
            print(resumen)
        else:
            print("⚠️ No se pudo construir el features final; revisá los lotes.")