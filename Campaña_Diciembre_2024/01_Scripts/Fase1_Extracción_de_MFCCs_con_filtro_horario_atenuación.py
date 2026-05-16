# ================================================================
# Fase 1 por campaña / por grabadora
# Extracción de MFCCs con filtro horario + atenuación de insectos
# ------------------------------------------------
# ¿Qué hace este script?
#   - Recorre todas las grabadoras ("sitios") de una campaña en el disco externo.
#   - Para cada sitio:
#       * Busca los archivos WAV dentro de la carpeta Data.
#       * Filtra los audios por horario (solo horas "de anfibios").
#       * Convierte el audio a mono y a una frecuencia fija (SR_TARGET).
#       * Aplica una atenuación adaptativa en una banda de frecuencias
#         donde suelen aparecer insectos (FMIN_INSECT–FMAX_INSECT),
#         reduciendo su energía sin eliminar completamente esa zona.
#       * Divide el audio en segmentos solapados de duración fija.
#       * Calcula MFCCs para cada segmento y guarda:
#             - archivo de origen, sitio, tiempo de inicio y fin,
#             - medias y desvíos de los MFCCs (mfcc_mean_*, mfcc_sd_*).
#       * Guarda los resultados por lotes en archivos .parquet temporales.
#   - Al final, concatena todos los lotes de cada sitio en un único
#     archivo de features por sitio y genera un resumen de cuántos
#     segmentos se obtuvieron.
#
# En resumen: prepara una tabla grande de segmentos con MFCCs por sitio,
# lista para usar en Fase 2 (UMAP + clustering).
# ================================================================

import re
import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
import librosa
from tqdm import tqdm

# ---------------- Configuración global ----------------
DURACION_SEGMENTO_SEG = 4
SOLAPAMIENTO_SEG = 2
NUM_MFCC = 20
TAMANO_BLOQUE = 100

SR_TARGET = 22050
N_FFT = 2048
HOP_LENGTH = 512

FMIN_INSECT = 4000      # insectos fuertes empiezan acá
FMAX_INSECT = 10000     # insectos moderados llegan hasta acá
BASE_GAIN_DB = -9.0     # atenuación mínima en toda la banda
MAX_EXTRA_DB = -6.0     # atenuación adicional si la banda domina
SMOOTH = 0.9            # suavizado temporal (esto ya está bien)

# =========================
# CAMPAÑA A PROCESAR
# =========================
NOMBRE_CAMPANIA = "Campaña diciembre 2024"

# Ruta base del HDD externo (ajustá la letra si cambia)
DISCO_EXTERNO = Path(r"D:\\")

# Carpeta de la campaña en el HDD externo
ruta_campania = DISCO_EXTERNO / NOMBRE_CAMPANIA

# Carpeta base de resultados en tu PC
BASE_RESULTADOS = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local") / "resultados_HDD_Seagate" / NOMBRE_CAMPANIA
BASE_RESULTADOS.mkdir(parents=True, exist_ok=True)

# Sufijo distintivo para esta corrida (MODIFICABLE)
SUFIJO_CORRIDA = "v2_horario18a06_insectos6a12"

def validar_entorno():
    if not ruta_campania.exists():
        raise RuntimeError(f"⚠️ No se encontró la campaña en el disco externo: {ruta_campania}")
    try:
        import pyarrow  # noqa: F401
    except Exception:
        print("⚠️ pyarrow no está disponible. Instalá: pip install pyarrow==15.0.2")


def extraer_hora_desde_nombre(nombre_archivo: str) -> int:
    base = Path(nombre_archivo).stem
    m = re.search(r"_(\d{6})$", base)
    if not m:
        raise ValueError(f"No se pudo extraer HHMMSS de: {nombre_archivo}")
    hhmmss = m.group(1)
    return int(hhmmss[0:2])


def es_hora_anfibios(hora: int) -> bool:
    return (hora >= 18) or (hora < 6)


def adaptive_attenuation(y, sr,
                         fmin=FMIN_INSECT, fmax=FMAX_INSECT,
                         base_gain_db=BASE_GAIN_DB, max_extra_db=MAX_EXTRA_DB,
                         n_fft=N_FFT, hop_length=HOP_LENGTH, smooth=SMOOTH):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
    mag, phase = np.abs(S), np.angle(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band_mask = (freqs >= fmin) & (freqs <= fmax)

    total_energy = (mag ** 2).sum(axis=0) + 1e-12
    band_energy = (mag[band_mask, :] ** 2).sum(axis=0) + 1e-12
    rel = band_energy / total_energy

    rel_norm = rel / (rel.max() + 1e-12)
    rel_smooth = np.zeros_like(rel_norm)
    for t in range(len(rel_norm)):
        if t == 0:
            rel_smooth[t] = rel_norm[t]
        else:
            rel_smooth[t] = smooth * rel_smooth[t-1] + (1 - smooth) * rel_norm[t]

    att_db = base_gain_db + (max_extra_db * rel_smooth)
    att_lin = 10.0 ** (att_db / 20.0)

    mag[band_mask, :] *= att_lin[np.newaxis, :]

    S_new = mag * np.exp(1j * phase)
    y_out = librosa.istft(S_new, hop_length=hop_length, window='hann')
    return y_out


def procesar_segmento(segmento, sr):
    try:
        mfccs = librosa.feature.mfcc(y=segmento.astype(np.float32), sr=sr, n_mfcc=NUM_MFCC)
        return np.hstack([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)])
    except Exception as e:
        print(f"[ERROR] procesar_segmento: {e}")
        return None


def procesar_archivo(ruta: Path, sitio: str):
    try:
        hora = extraer_hora_desde_nombre(ruta.name)
        if not es_hora_anfibios(hora):
            return None

        audio, sr = sf.read(ruta)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != SR_TARGET:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR_TARGET)
            sr = SR_TARGET

        duracion = len(audio) / sr
        if duracion < DURACION_SEGMENTO_SEG:
            return None

        audio_filt = adaptive_attenuation(audio, sr)

        frame_length = int(DURACION_SEGMENTO_SEG * sr)
        hop_length = int(SOLAPAMIENTO_SEG * sr)
        frames = librosa.util.frame(audio_filt, frame_length=frame_length, hop_length=hop_length).T

        resultados = []
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


def correr_fase1_para_sitio(sitio: str):
    print(f"\n==============================")
    print(f"Procesando grabadora: {sitio}")
    print(f"Campaña: {NOMBRE_CAMPANIA}")
    print(f"==============================")

    # Carpeta de resultados específica para esta grabadora
    ruta_salida_sitio = BASE_RESULTADOS / sitio
    ruta_salida_sitio.mkdir(parents=True, exist_ok=True)

    ruta_temporales = ruta_salida_sitio / f"temporales_fase1_{sitio}_{SUFIJO_CORRIDA}"
    ruta_temporales.mkdir(parents=True, exist_ok=True)

    ruta_features_final = ruta_salida_sitio / f"features_{sitio}_{SUFIJO_CORRIDA}.parquet"
    ruta_resumen = ruta_salida_sitio / f"resumen_segmentos_por_sitio_{sitio}_{SUFIJO_CORRIDA}.csv"

    carpeta = ruta_campania / sitio
    data_dir = carpeta / "Data"
    if not data_dir.exists():
        print(f"⚠️ No se encontró la carpeta 'Data' en {carpeta}, se salta esta grabadora.")
        return

    archivos = list(data_dir.rglob("*.wav"))
    print(f"{sitio}: {len(archivos)} archivos .wav encontrados.")
    if len(archivos) == 0:
        return

    num_bloques = max(1, len(archivos) // TAMANO_BLOQUE)
    bloques = np.array_split(archivos, num_bloques)

    for i, bloque in enumerate(bloques):
        nombre_lote = f"lote_{sitio}_{SUFIJO_CORRIDA}_{i+1:03d}.parquet"
        ruta_lote = ruta_temporales / nombre_lote

        if ruta_lote.exists():
            print(f"⏩ {sitio} Lote {i+1}/{len(bloques)} ya procesado, se salta.")
            continue

        print(f"📦 {sitio} Lote {i+1}/{len(bloques)} con {len(bloque)} archivos")
        resultados_bloque = []
        for r in tqdm(bloque, desc=f"{sitio} Lote {i+1}", unit="arch"):
            df = procesar_archivo(r, sitio)
            if df is not None:
                resultados_bloque.append(df)

        if resultados_bloque:
            df_lote = pd.concat(resultados_bloque, ignore_index=True)
            df_lote.to_parquet(ruta_lote, index=False)
            print(f"✅ Lote guardado: {nombre_lote}")
        else:
            print(f"⚠️ Lote vacío: {nombre_lote}")

    lotes = sorted(ruta_temporales.glob("lote_*.parquet"))
    if len(lotes) == 0:
        print(f"⚠️ {sitio}: no se encontraron lotes para concatenar.")
        return

    print(f"🔗 {sitio}: concatenando {len(lotes)} lotes...")
    dfs = []
    for p in lotes:
        try:
            dfs.append(pd.read_parquet(p))
        except Exception as e:
            print(f"[ERROR] al leer {p.name}: {e}")

    if dfs:
        datos = pd.concat(dfs, ignore_index=True)
        datos.to_parquet(ruta_features_final, index=False)
        print(f"💾 {sitio}: guardado final en {ruta_features_final} con {len(datos)} segmentos.")

        resumen = datos.groupby("sitio").size().reset_index(name="segmentos")
        resumen.to_csv(ruta_resumen, index=False)
        print(f"📊 Resumen por sitio {sitio}:")
        print(resumen)
    else:
        print(f"⚠️ {sitio}: no se pudo construir el features final.")


if __name__ == "__main__":
    validar_entorno()

    # Detectar todas las grabadoras en el disco externo (campaña)
    grabadoras = [p.name for p in ruta_campania.iterdir() if p.is_dir()]
    grabadoras = sorted(grabadoras)

    print("\nGrabadoras detectadas en la campaña:")
    for g in grabadoras:
        print(" -", g)

    # Detectar cuáles faltan procesar:
    # condición: tiene Data en el disco externo PERO no tiene features finales en BASE_RESULTADOS
    sitios_pendientes = []
    for sitio in grabadoras:
        carpeta = ruta_campania / sitio
        data_dir = carpeta / "Data"

        ruta_salida_sitio = BASE_RESULTADOS / sitio
        ruta_features_final = ruta_salida_sitio / f"features_{sitio}_{SUFIJO_CORRIDA}.parquet"

        if data_dir.exists() and not ruta_features_final.exists():
            sitios_pendientes.append(sitio)

    print("\nGrabadoras pendientes de procesar (con Data y sin features finales):")
    if not sitios_pendientes:
        print(" - Ninguna. Todo está procesado para esta campaña.")
    else:
        for s in sitios_pendientes:
            print(" -", s)

        # Procesar solo las pendientes
        for sitio in sitios_pendientes:
            correr_fase1_para_sitio(sitio)