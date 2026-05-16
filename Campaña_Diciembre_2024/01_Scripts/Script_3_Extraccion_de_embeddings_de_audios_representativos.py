# ================================================================
# Script 3: Extracción de embeddings de audios representativos
# (con filtro de insectos antes de YAMNet)
# ------------------------------------------------
# DESCRIPCIÓN BREVE
#
# Este script busca los audios representativos generados en:
#   - Fase 3: figuras_inspeccion_clusters/<campaña>/<sitio>/*.wav
#   - Fase 4.3: figuras_inspeccion_subclusters/<campaña>/<sitio>/audios_representativos/*.wav
#
# Para cada .wav:
#   - lo pasa por un filtro de insectos (mismo criterio que MFCCs),
#   - lo pasa a 16 kHz mono,
#   - lo divide en ventanas de 1.5 s con solapamiento de 0.75 s,
#   - extrae embeddings con YAMNet y promedia,
#   - guarda un vector por archivo (firma acústica).
#
# El resultado es un .parquet con un embedding por archivo
# representativo, listo para compararse con la base del Script 2.
# ================================================================

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import tensorflow as tf
import tensorflow_hub as hub

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- CONFIGURACIÓN ----------------

NOMBRE_CAMPANIA = "Campaña diciembre 2024"

BASE_DIR = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local")

RUTA_REP_FASE3 = BASE_DIR / "figuras_inspeccion_clusters" / NOMBRE_CAMPANIA
RUTA_REP_FASE4_3 = BASE_DIR / "figuras_inspeccion_subclusters" / NOMBRE_CAMPANIA

RUTA_SALIDA = BASE_DIR / "embeddings_representativos"
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

SR_YAMNET = 16000
VENTANA_SEG = 1.5
SOLAPAMIENTO_SEG = 0.75

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"


# ---------------- FILTRO DE INSECTOS (REAL) ----------------

def filtrar_insectos(y: np.ndarray, sr: int) -> np.ndarray:
    FMIN_INSECT = 4000
    FMAX_INSECT = 10000
    BASE_GAIN_DB = -9.0
    MAX_EXTRA_DB = -6.0
    SMOOTH = 0.9

    S = librosa.stft(y, n_fft=2048, hop_length=512)
    mag, phase = librosa.magphase(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    idx_insect = np.where((freqs >= FMIN_INSECT) & (freqs <= FMAX_INSECT))[0]

    energia_total = np.sum(mag, axis=0) + 1e-12
    energia_insectos = np.sum(mag[idx_insect, :], axis=0)

    propor = energia_insectos / energia_total

    extra = MAX_EXTRA_DB * propor
    gain_db = BASE_GAIN_DB + extra

    gain_db_smooth = np.zeros_like(gain_db)
    gain_db_smooth[0] = gain_db[0]
    for i in range(1, len(gain_db)):
        gain_db_smooth[i] = SMOOTH * gain_db_smooth[i - 1] + (1 - SMOOTH) * gain_db[i]

    gain_lin = librosa.db_to_amplitude(gain_db_smooth)
    mag_filt = mag * gain_lin[np.newaxis, :]

    S_filt = mag_filt * phase
    y_filt = librosa.istft(S_filt, hop_length=512)

    return y_filt.astype(np.float32)


# ---------------- FUNCIONES YAMNET ----------------

def cargar_yamnet_model():
    print("Cargando modelo YAMNet desde TensorFlow Hub...")
    model = hub.load(YAMNET_HANDLE)
    return model


def resample_audio(y, sr_orig, sr_target):
    if sr_orig == sr_target:
        return y
    return librosa.resample(y, orig_sr=sr_orig, target_sr=sr_target)


def extraer_embeddings_archivo(ruta_wav: Path, yamnet_model):
    try:
        y, sr = sf.read(str(ruta_wav))
    except Exception as e:
        print(f"⚠️ Error al leer {ruta_wav}: {e}")
        return None

    if y.ndim > 1:
        y = y.mean(axis=1)

    # Filtro de insectos
    y = filtrar_insectos(y, sr)

    # Resampleo a 16 kHz
    y = resample_audio(y, sr, SR_YAMNET)
    sr = SR_YAMNET

    dur_total = len(y) / sr
    if dur_total < VENTANA_SEG:
        return None

    paso = int((VENTANA_SEG - SOLAPAMIENTO_SEG) * sr)
    tam_ventana = int(VENTANA_SEG * sr)

    embeddings = []
    for inicio in range(0, len(y) - tam_ventana + 1, paso):
        segmento = y[inicio:inicio + tam_ventana].astype(np.float32)
        _, emb, _ = yamnet_model(segmento)
        emb_np = emb.numpy()
        emb_mean = emb_np.mean(axis=0)
        embeddings.append(emb_mean)

    if not embeddings:
        return None

    embeddings = np.vstack(embeddings)
    emb_promedio = embeddings.mean(axis=0)

    return emb_promedio, dur_total, embeddings.shape[0]


def recolectar_archivos_representativos():
    archivos = []
    if RUTA_REP_FASE3.exists():
        archivos += list(RUTA_REP_FASE3.rglob("*.wav"))
    if RUTA_REP_FASE4_3.exists():
        archivos += list(RUTA_REP_FASE4_3.rglob("*.wav"))
    return sorted(set(archivos))


def main():
    archivos = recolectar_archivos_representativos()
    if not archivos:
        print("⚠️ No se encontraron archivos representativos.")
        return

    print("\n================================================")
    print("Extracción de embeddings de audios representativos (con filtro de insectos)")
    print("Total de archivos detectados:", len(archivos))
    print("================================================\n")

    yamnet_model = cargar_yamnet_model()

    registros = []

    for ruta_wav in archivos:
        partes = ruta_wav.relative_to(BASE_DIR).parts

        sitio = "desconocido"
        grupo = ruta_wav.parent.name

        if len(partes) >= 3 and partes[0] == "figuras_inspeccion_clusters":
            sitio = partes[2]
            grupo = "cluster"
        elif len(partes) >= 4 and partes[0] == "figuras_inspeccion_subclusters":
            sitio = partes[2]
            if "audios_representativos" in partes:
                grupo = "subcluster_representativo"
            else:
                grupo = "subcluster_otro"

        emb_info = extraer_embeddings_archivo(ruta_wav, yamnet_model)
        if emb_info is None:
            print(f"⚠️ No se pudo extraer embedding de: {ruta_wav}")
            continue

        emb_vec, dur_total, n_ventanas = emb_info

        registro = {
            "sitio": sitio,
            "grupo": grupo,
            "archivo": ruta_wav.name,
            "ruta_relativa_desde_BASE_DIR": str(ruta_wav.relative_to(BASE_DIR)),
            "duracion_s": dur_total,
            "n_ventanas": n_ventanas,
        }

        for i, val in enumerate(emb_vec):
            registro[f"emb_{i}"] = float(val)

        registros.append(registro)

    if not registros:
        print("⚠️ No se generaron embeddings.")
        return

    df = pd.DataFrame(registros)
    out_parquet = RUTA_SALIDA / "embeddings_representativos_yamnet.parquet"
    df.to_parquet(out_parquet, index=False)

    print("\n✔️ Embeddings guardados en:", out_parquet)
    print("   Total de registros:", len(df))


if __name__ == "__main__":
    main()