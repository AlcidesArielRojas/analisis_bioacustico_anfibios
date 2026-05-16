# ================================================================
# Script 2: Extracción de embeddings de referencia por especie
# ------------------------------------------------
# DESCRIPCIÓN BREVE
#
# Este script recorre toda la base de datos de anfibios en WAV
# (BD_anfibios_wav). Para cada archivo:
#   - lo convierte a 16 kHz mono (formato YAMNet),
#   - lo divide en ventanas cortas,
#   - extrae embeddings con YAMNet,
#   - promedia los embeddings del archivo.
#
# Caso especial: "Cantos Paraguay"
#   Esta carpeta contiene audios de muchas especies mezcladas.
#   El script detecta automáticamente la especie REAL a partir
#   del nombre del archivo (ej: "11-A11 - Dendropsophus minutus.wav")
#   y asigna correctamente la especie, evitando que aparezca
#   "Cantos Paraguay" como especie falsa.
#
# El resultado final es un archivo parquet con las huellas acústicas
# de referencia por especie, listo para el matching automático.
# ================================================================

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import re

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- CONFIGURACIÓN ----------------

BASE_DIR = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local")

RUTA_BD_WAV = BASE_DIR / "BD_anfibios_wav"
RUTA_SALIDA = BASE_DIR / "embeddings_BD_anfibios"
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

SR_YAMNET = 16000
VENTANA_SEG = 1.5
SOLAPAMIENTO_SEG = 0.75

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"


# ------------------------------------------------
# Función para extraer especie desde nombres como:
# "11-A11 - Dendropsophus minutus.wav"
# ------------------------------------------------
def extraer_especie_desde_nombre(nombre_archivo: str) -> str | None:
    """
    Extrae la especie real desde nombres del tipo:
    "11-A11 - Dendropsophus minutus.wav"
    Devuelve: "Dendropsophus_minutus"
    """
    base = Path(nombre_archivo).stem

    # Buscar patrón después del guion y espacio: " - "
    if " - " in base:
        especie_raw = base.split(" - ", 1)[1].strip()
    else:
        # fallback: buscar dos palabras latinas consecutivas
        m = re.search(r"([A-Z][a-z]+)\s+([a-z]+)", base)
        if not m:
            return None
        especie_raw = f"{m.group(1)} {m.group(2)}"

    especie = especie_raw.replace(" ", "_")
    return especie


# ------------------------------------------------
# Cargar modelo YAMNet
# ------------------------------------------------
def cargar_yamnet_model():
    print("Cargando modelo YAMNet desde TensorFlow Hub...")
    model = hub.load(YAMNET_HANDLE)
    return model


# ------------------------------------------------
# Resampleo
# ------------------------------------------------
def resample_audio(y, sr_orig, sr_target):
    if sr_orig == sr_target:
        return y
    return librosa.resample(y, orig_sr=sr_orig, target_sr=sr_target)


# ------------------------------------------------
# Extraer embeddings de un archivo WAV
# ------------------------------------------------
def extraer_embeddings_archivo(ruta_wav: Path, yamnet_model):
    try:
        y, sr = sf.read(str(ruta_wav))
    except Exception as e:
        print(f"⚠️ Error al leer {ruta_wav}: {e}")
        return None

    if y.ndim > 1:
        y = y.mean(axis=1)

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


# ------------------------------------------------
# PROCESAMIENTO PRINCIPAL
# ------------------------------------------------
def main():
    if not RUTA_BD_WAV.exists():
        print(f"⚠️ No se encontró la carpeta BD_anfibios_wav: {RUTA_BD_WAV}")
        return

    yamnet_model = cargar_yamnet_model()

    print("\n================================================")
    print("Extracción de embeddings de referencia (YAMNet)")
    print("================================================\n")

    carpetas = [p for p in RUTA_BD_WAV.iterdir() if p.is_dir()]
    carpetas = sorted(carpetas, key=lambda p: p.name)

    registros = []

    for carpeta in carpetas:
        nombre_carpeta = carpeta.name

        # ------------------------------------------------
        # CASO ESPECIAL: "Cantos Paraguay"
        # ------------------------------------------------
        if nombre_carpeta.lower().replace(" ", "") == "cantosparaguay":
            print(f"\nProcesando carpeta especial: {nombre_carpeta}")

            archivos = sorted(carpeta.glob("*.wav"))
            for ruta_wav in archivos:
                especie = extraer_especie_desde_nombre(ruta_wav.name)
                if especie is None:
                    print(f"⚠️ No se pudo extraer especie de: {ruta_wav.name}")
                    continue

                emb_info = extraer_embeddings_archivo(ruta_wav, yamnet_model)
                if emb_info is None:
                    print(f"⚠️ No se pudo extraer embedding de: {ruta_wav.name}")
                    continue

                emb_vec, dur_total, n_ventanas = emb_info

                registro = {
                    "especie": especie,
                    "archivo": ruta_wav.name,
                    "ruta_relativa": f"Cantos Paraguay/{ruta_wav.name}",
                    "duracion_s": dur_total,
                    "n_ventanas": n_ventanas,
                }

                for i, val in enumerate(emb_vec):
                    registro[f"emb_{i}"] = float(val)

                registros.append(registro)

            continue  # saltar a la siguiente carpeta


        # ------------------------------------------------
        # CASO NORMAL: carpetas de especies reales
        # ------------------------------------------------
        especie = nombre_carpeta.replace(" ", "_")
        archivos = sorted(carpeta.glob("*.wav"))

        if not archivos:
            print(f"⚠️ Sin .wav para especie: {especie}")
            continue

        print(f"\nEspecie: {especie} | {len(archivos)} archivos")

        for ruta_wav in archivos:
            emb_info = extraer_embeddings_archivo(ruta_wav, yamnet_model)
            if emb_info is None:
                print(f"⚠️ No se pudo extraer embedding de: {ruta_wav.name}")
                continue

            emb_vec, dur_total, n_ventanas = emb_info

            registro = {
                "especie": especie,
                "archivo": ruta_wav.name,
                "ruta_relativa": str(ruta_wav.relative_to(RUTA_BD_WAV)),
                "duracion_s": dur_total,
                "n_ventanas": n_ventanas,
            }

            for i, val in enumerate(emb_vec):
                registro[f"emb_{i}"] = float(val)

            registros.append(registro)

    # ------------------------------------------------
    # GUARDAR RESULTADOS
    # ------------------------------------------------
    if not registros:
        print("⚠️ No se generaron embeddings.")
        return

    df = pd.DataFrame(registros)
    out_parquet = RUTA_SALIDA / "embeddings_BD_anfibios_yamnet.parquet"
    df.to_parquet(out_parquet, index=False)

    print("\n✔️ Embeddings guardados en:", out_parquet)
    print("   Total de registros:", len(df))


if __name__ == "__main__":
    main()