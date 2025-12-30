# ================================================================
# Fase 1 Benchmark: Extracción de MFCCs para sitio PA-41Tapyta
# Adaptado por Alcides Rojas
# ================================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
from scipy import signal
import librosa
from tqdm import tqdm

# --- Configuración ---
DURACION_SEGMENTO_SEG = 4
SOLAPAMIENTO_SEG = 2
LIMITE_INFERIOR_HZ = 700
LIMITE_SUPERIOR_HZ = 2500
NUM_MFCC = 20
TAMANO_BLOQUE = 100

# Sitio específico para benchmark
TARGET_SITIO = "PA-41Tapyta"

# Rutas
ruta_base_externa = Path(r"D:\\")
ruta_salida = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local\resultados")
ruta_salida.mkdir(exist_ok=True)

ruta_temporales = ruta_salida / f"temporales_fase1_{TARGET_SITIO}"
ruta_temporales.mkdir(exist_ok=True)
ruta_features_final = ruta_salida / f"features_{TARGET_SITIO}.parquet"
ruta_resumen = ruta_salida / f"resumen_segmentos_por_sitio_{TARGET_SITIO}.csv"

# --- Validación de entorno ---
def validar_entorno():
    if not ruta_base_externa.exists():
        raise RuntimeError(f"⚠️ Disco externo no encontrado: {ruta_base_externa}")
    try:
        import pyarrow  # noqa: F401
    except Exception:
        print("⚠️ pyarrow no está disponible. Instalá: pip install pyarrow==15.0.2")

# --- Procesamiento de segmento ---
def procesar_segmento(segmento, sr):
    try:
        nyquist = 0.5 * sr
        b, a = signal.butter(4, [LIMITE_INFERIOR_HZ/nyquist, LIMITE_SUPERIOR_HZ/nyquist], btype='band')
        filtrado = signal.filtfilt(b, a, segmento).astype(np.float32)
        # Eliminamos reducción de ruido para conservar más información
        procesado = filtrado
        mfccs = librosa.feature.mfcc(y=procesado, sr=sr, n_mfcc=NUM_MFCC)
        return np.hstack([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)])
    except Exception as e:
        print(f"[ERROR] procesar_segmento: {e}")
        return None

# --- Procesamiento de archivo ---
def procesar_archivo(ruta: Path):
    try:
        audio, sr = sf.read(ruta)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        duracion = len(audio) / sr
        if duracion < DURACION_SEGMENTO_SEG:
            return None

        frame_length = int(DURACION_SEGMENTO_SEG * sr)
        hop_length = int(SOLAPAMIENTO_SEG * sr)
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length).T

        resultados = []
        sitio = TARGET_SITIO

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
    validar_entorno()
    carpeta = ruta_base_externa / TARGET_SITIO
    data_dir = carpeta / 'Data'
    if not data_dir.exists():
        raise RuntimeError(f"No se encontró la carpeta 'Data' en {carpeta}")

    archivos = list(data_dir.rglob('*.wav'))
    print(f"Procesando sitio: {TARGET_SITIO} ({len(archivos)} archivos)")
    if len(archivos) == 0:
        raise RuntimeError("No se encontraron archivos .wav para procesar.")

    num_bloques = max(1, len(archivos) // TAMANO_BLOQUE)
    bloques = np.array_split(archivos, num_bloques)

    for i, bloque in enumerate(bloques):
        nombre_lote = f"lote_{TARGET_SITIO}_{i+1:03d}.parquet"
        ruta_lote = ruta_temporales / nombre_lote

        if ruta_lote.exists():
            print(f"⏩ Lote {i+1}/{len(bloques)} ya procesado, se salta.")
            continue

        print(f"📦 Lote {i+1}/{len(bloques)} con {len(bloque)} archivos")
        resultados_bloque = []
        for r in tqdm(bloque, desc=f"{TARGET_SITIO} Lote {i+1}", unit="arch"):
            df = procesar_archivo(r)
            if df is not None:
                resultados_bloque.append(df)

        if resultados_bloque:
            df_lote = pd.concat(resultados_bloque, ignore_index=True)
            df_lote.to_parquet(ruta_lote, index=False)
            print(f"✅ Lote guardado: {nombre_lote}")
        else:
            print(f"⚠️ Lote {i+1} sin resultados")

    # Concatenación final
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
