# ================================================================
# Fase 1: Extracción de características de audio (MFCCs) optimizada
# Autor: Alcides Rojas
# Fecha: 2025-12-16
# Correcciones:
# - Sin paralelismo (secual: estabilidad y menos I/O conflictivo)
# - Guardado incremental por lote en .parquet
# - Detección de lotes ya procesados y reanudación segura
# - Recorrido de todas las carpetas "BO*" con subcarpeta "Data"
# - Concatenación final desde temporales
# ================================================================

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
from scipy import signal
import librosa
# noisereduce opcional; desactivado en este flujo por estabilidad
# import noisereduce as nr
from tqdm import tqdm

# --- Configuración ---
DURACION_SEGMENTO_SEG = 4
SOLAPAMIENTO_SEG = 2
LIMITE_INFERIOR_HZ = 700
LIMITE_SUPERIOR_HZ = 2500
NUM_MFCC = 20
TAMANO_BLOQUE = 200  # cantidad de archivos por lote

# Disco externo y salidas
ruta_base_externa = Path(r"D:\\")  # Ajusta si tu disco externo tiene otra letra
ruta_raiz_repo = Path(r"C:\Users\Alcides\Proyecto_Paisajes_Sonoros_Repositorio_Local")
ruta_salida = ruta_raiz_repo / "resultados"
ruta_temporales = ruta_salida / "temporales_fase1"
ruta_features_final = ruta_salida / "features.parquet"
ruta_resumen = ruta_salida / "resumen_segmentos_por_sitio.csv"

ruta_salida.mkdir(parents=True, exist_ok=True)
ruta_temporales.mkdir(parents=True, exist_ok=True)

# --- Validación de entorno y dependencias ---
def validar_entorno():
    if not ruta_base_externa.exists():
        raise RuntimeError(f"Disco externo no encontrado: {ruta_base_externa}")
    try:
        import pyarrow  # noqa: F401
    except Exception:
        print("pyarrow no está disponible. Instalá: pip install pyarrow==15.0.2")
        print("Alternativamente, cambia a CSV con to_csv (menos eficiente).")

# --- Procesamiento de un segmento ---
def procesar_segmento(segmento: np.ndarray, sr: int) -> np.ndarray | None:
    """
    Aplica filtrado banda y calcula MFCCs (medias y desviaciones).
    Devuelve un vector de longitud 40 (20 mean + 20 sd) o None si falla.
    """
    try:
        nyquist = 0.5 * sr
        low = LIMITE_INFERIOR_HZ / nyquist
        high = LIMITE_SUPERIOR_HZ / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtrado = signal.filtfilt(b, a, segmento).astype(np.float32)
        # reducido = nr.reduce_noise(y=filtrado, sr=sr)  # desactivado por estabilidad
        mfccs = librosa.feature.mfcc(y=filtrado, sr=sr, n_mfcc=NUM_MFCC)
        return np.hstack([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)])
    except Exception as e:
        print(f"[ERROR] procesar_segmento: {e}")
        return None

# --- Procesamiento de un archivo ---
def procesar_archivo(ruta: Path) -> pd.DataFrame | None:
    """
    Procesa un archivo .wav en ventanas solapadas de 4s, 2s hop.
    Devuelve DataFrame con metadata y 40 features MFCC por segmento.
    """
    try:
        if not ruta.exists() or ruta.stat().st_size == 0:
            print(f"[WARN] Archivo vacío o inexistente: {ruta.name}")
            return None

        audio, sr = sf.read(ruta)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # a mono

        duracion = len(audio) / sr
        if duracion < DURACION_SEGMENTO_SEG:
            print(f"[WARN] Audio demasiado corto (<{DURACION_SEGMENTO_SEG}s): {ruta.name}")
            return None

        frame_length = int(DURACION_SEGMENTO_SEG * sr)
        hop_length = int(SOLAPAMIENTO_SEG * sr)

        # Determinar el 'sitio' según estructura (.../BO-xx/Data/archivo.wav)
        partes = ruta.parts
        sitio = None
        if "Data" in partes:
            idx = partes.index("Data")
            sitio = partes[idx - 1] if idx > 0 else ruta.parent.name
        else:
            # fallback si no hay 'Data' en la ruta
            sitio = ruta.parent.name

        archivo_rel = f"{sitio}/{ruta.name}"
        nombres = [f"mfcc_mean_{j+1}" for j in range(NUM_MFCC)] + \
                  [f"mfcc_sd_{j+1}" for j in range(NUM_MFCC)]

        filas = []
        for inicio in range(0, len(audio) - frame_length + 1, hop_length):
            seg = audio[inicio:inicio + frame_length]
            vec = procesar_segmento(seg, sr)
            if vec is not None:
                info = {
                    "archivo_origen": archivo_rel,
                    "sitio": sitio,
                    "tiempo_inicio": round(inicio / sr, 2),
                    "tiempo_fin": round((inicio + frame_length) / sr, 2),
                }
                filas.append({**info, **dict(zip(nombres, vec))})

        return pd.DataFrame(filas) if filas else None

    except Exception as e:
        print(f"[ERROR] procesar_archivo ({ruta.name}): {e}")
        return None

# --- Utilidad: listar sitios "BO*" con carpeta Data ---
def listar_sitios_con_data(base: Path) -> list[Path]:
    sitios = []
    try:
        for p in base.iterdir():
            if p.is_dir() and p.name.startswith("BO"):
                data_dir = p / "Data"
                if data_dir.exists():
                    sitios.append(p)
    except Exception as e:
        print(f"[ERROR] listando sitios: {e}")
    return sitios

# --- Orquestación con guardado incremental por lote ---
def correr_fase1():
    validar_entorno()

    sitios = listar_sitios_con_data(ruta_base_externa)
    if not sitios:
        print("No se encontraron carpetas 'BO*' con subcarpeta 'Data' en el disco externo.")
        return

    print(f"Sitios detectados: {[s.name for s in sitios]}")
    for carpeta in sitios:
        data_dir = carpeta / "Data"
        archivos = sorted(data_dir.rglob("*.wav"))
        if not archivos:
            print(f"[WARN] Sin .wav en: {carpeta.name}/Data")
            continue

        print(f"Procesando sitio: {carpeta.name} ({len(archivos)} archivos)")
        # Dividir en bloques/lotes
        bloques = np.array_split(np.array(archivos), max(1, len(archivos) // TAMANO_BLOQUE))

        for i, bloque in enumerate(bloques, start=1):
            nombre_lote = f"{carpeta.name}_lote_{i:03d}.parquet"
            ruta_lote = ruta_temporales / nombre_lote

            if ruta_lote.exists():
                print(f"Lote ya procesado, salto: {nombre_lote}")
                continue

            print(f"Lote {i}/{len(bloques)} con {len(bloque)} archivos")
            resultados_bloque = []
            # Procesamiento secuencial para estabilidad de I/O
            for ruta in tqdm(bloque, desc=f"{carpeta.name} Lote {i}", unit="arch"):
                df = procesar_archivo(Path(ruta))
                if df is not None and not df.empty:
                    resultados_bloque.append(df)

            if resultados_bloque:
                df_lote = pd.concat(resultados_bloque, ignore_index=True)
                try:
                    df_lote.to_parquet(ruta_lote, index=False)  # engine='pyarrow' automático
                    print(f"Lote guardado: {nombre_lote} ({len(df_lote)} filas)")
                except Exception as e:
                    print(f"No se pudo guardar {nombre_lote}: {e}")
            else:
                print(f"Lote {i} sin resultados")

    # Concatenación final desde temporales
    lotes = sorted(ruta_temporales.glob("BO*_lote_*.parquet"))
    if not lotes:
        print("No hay lotes temporales para concatenar.")
        return

    print(f"Concatenando {len(lotes)} lotes en {ruta_features_final.name}…")
    try:
        datos = pd.concat((pd.read_parquet(p) for p in lotes), ignore_index=True)
        datos.to_parquet(ruta_features_final, index=False)
        print(f"Final guardado: {ruta_features_final} ({len(datos)} filas)")

        resumen = datos.groupby("sitio").size().reset_index(name="segmentos")
        resumen.to_csv(ruta_resumen, index=False)
        print("Resumen por sitio:")
        print(resumen)
    except Exception as e:
        print(f"No se pudo construir el features final; revisá los lotes. Error: {e}")

if __name__ == "__main__":
    correr_fase1()