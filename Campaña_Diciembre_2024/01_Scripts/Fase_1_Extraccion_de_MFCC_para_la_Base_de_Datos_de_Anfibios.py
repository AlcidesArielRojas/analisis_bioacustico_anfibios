# ================================================================
# Fase 1 — Extracción de MFCC para la Base de Datos de Anfibios (BD)
# ------------------------------------------------
# Procesa todos los audios .wav de la base de datos de anfibios, aplicando la 
# misma lógica de preprocesamiento usada en la campaña (resampleo, atenuación de 
# insectos, segmentación y extracción de MFCC). Para cada segmento generado, guarda 
# la especie, el archivo de origen y los tiempos asociados, permitiendo rastrear cada 
# audio de la BD dentro del espacio UMAP de la campaña y asignarlo posteriormente 
# a los clusters y especies correspondientes.
# ================================================================


import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
import librosa
from tqdm import tqdm

# ============================================================
# CONFIGURACIÓN GLOBAL (idéntica a Fase 1 de campaña)
# ============================================================

DURACION_SEGMENTO_SEG = 4
SOLAPAMIENTO_SEG = 2
NUM_MFCC = 20

SR_TARGET = 22050
N_FFT = 2048
HOP_LENGTH = 512

# Filtro de insectos (igual que campaña)
FMIN_INSECT = 4000
FMAX_INSECT = 10000
BASE_GAIN_DB = -9.0
MAX_EXTRA_DB = -6.0
SMOOTH = 0.9

# ============================================================
# RUTAS
# ============================================================

BASE_DIR = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local")

# Carpeta donde están los WAV de la BD (estructura por especie)
RUTA_BD_WAV = BASE_DIR / "BD_anfibios_wav"

# Carpeta donde guardaremos los MFCC de la BD
RUTA_SALIDA_BD = BASE_DIR / "proyeccion_BD_fase1_5"
RUTA_SALIDA_BD.mkdir(parents=True, exist_ok=True)

# Archivo final de MFCC de BD
RUTA_FEATURES_BD = RUTA_SALIDA_BD / "features_BD_anfibios.parquet"

# ============================================================
# NORMALIZACIÓN TAXONÓMICA
# ============================================================

# Diccionario de normalización: nombre crudo -> nombre moderno corregido
NORMALIZAR_ESPECIE = {
    # Dendropsophus / Dendrosophus
    "Dendrosophus minutus": "Dendropsophus minutus",
    "Dendrosophus nanus": "Dendropsophus nanus",
    "Dendrosophus sanborni": "Dendropsophus sanborni",

    # Hypsiboas -> Boana (taxonomía moderna)
    "Hypsiboas albopunctatus": "Boana albopunctata",
    "Hypsiboas caingua": "Boana caingua",
    "Hypsiboas curupi": "Boana curupi",
    "Hypsiboas faber": "Boana faber",
    "Hypsiboas pulchellus": "Boana pulchella",
    "Hypsiboas raniceps": "Boana raniceps",

    # Ololygon -> Scinax
    "Ololygon berthae": "Scinax berthae",

    # Itapotihyla
    "Itapotihyla langsdorfii": "Itapotihyla langsdorffii",

    # Melanophryniscus
    "Melanophrynisus atroluteus": "Melanophryniscus atroluteus",

    # Physalaemus
    "Physlaemus fernandezae": "Physalaemus fernandezae",

    # Pseudopaludicola
    "Pseudaludicola falcipes": "Pseudopaludicola falcipes",
}

def normalizar_nombre_especie(nombre: str) -> str:
    """
    Aplica normalización taxonómica:
    - corrige errores ortográficos y sinónimos según diccionario
    - mantiene 'cf.' y 'sp' como están (no se unifican)
    """
    nombre = nombre.strip()
    return NORMALIZAR_ESPECIE.get(nombre, nombre)


def especie_a_codigo(nombre: str) -> str:
    """
    Convierte 'Scinax fuscovarius' -> 'Sci_fuscovarius'
    Maneja casos con 'cf.' y 'sp' sin unificarlos.
    """
    partes = nombre.split()

    if len(partes) == 2:
        gen, esp = partes
        return f"{gen[:3].capitalize()}_{esp}"

    # Ej: 'Odontophrynus cf. americanus' -> 'Odo_cf_americanus'
    if len(partes) == 3 and partes[1] in {"cf.", "cf"}:
        gen, tag, esp = partes
        tag_clean = tag.replace(".", "")
        return f"{gen[:3].capitalize()}_{tag_clean}_{esp}"

    # Ej: 'Pithecopus sp' -> 'Pit_sp'
    if len(partes) == 2 and partes[1] in {"sp", "sp."}:
        gen, tag = partes
        tag_clean = tag.replace(".", "")
        return f"{gen[:3].capitalize()}_{tag_clean}"

    # Caso raro: más de 3 palabras, fallback
    return nombre.replace(" ", "_")


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def adaptive_attenuation(y, sr):
    """Misma atenuación de insectos que en campaña."""
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann')
    mag, phase = np.abs(S), np.angle(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    band_mask = (freqs >= FMIN_INSECT) & (freqs <= FMAX_INSECT)

    total_energy = (mag ** 2).sum(axis=0) + 1e-12
    band_energy = (mag[band_mask, :] ** 2).sum(axis=0) + 1e-12
    rel = band_energy / total_energy

    rel_norm = rel / (rel.max() + 1e-12)
    rel_smooth = np.zeros_like(rel_norm)
    for t in range(len(rel_norm)):
        if t == 0:
            rel_smooth[t] = rel_norm[t]
        else:
            rel_smooth[t] = SMOOTH * rel_smooth[t-1] + (1 - SMOOTH) * rel_norm[t]

    att_db = BASE_GAIN_DB + (MAX_EXTRA_DB * rel_smooth)
    att_lin = 10.0 ** (att_db / 20.0)

    mag[band_mask, :] *= att_lin[np.newaxis, :]
    S_new = mag * np.exp(1j * phase)
    return librosa.istft(S_new, hop_length=HOP_LENGTH, window='hann')


def procesar_segmento(seg, sr):
    try:
        mfccs = librosa.feature.mfcc(y=seg.astype(np.float32), sr=sr, n_mfcc=NUM_MFCC)
        return np.hstack([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)])
    except:
        return None


def procesar_archivo_bd(ruta_wav: Path, especie_codigo: str):
    """Procesa un archivo WAV de la BD y devuelve un DF con segmentos MFCC."""
    try:
        audio, sr = sf.read(ruta_wav)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != SR_TARGET:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR_TARGET)
            sr = SR_TARGET

        if len(audio) < DURACION_SEGMENTO_SEG * sr:
            return None

        # Atenuación de insectos (igual que campaña)
        audio_filt = adaptive_attenuation(audio, sr)

        frame_length = int(DURACION_SEGMENTO_SEG * sr)
        hop = int(SOLAPAMIENTO_SEG * sr)
        frames = librosa.util.frame(audio_filt, frame_length=frame_length, hop_length=hop).T

        filas = []
        for i, seg in enumerate(frames):
            vec = procesar_segmento(seg, sr)
            if vec is None:
                continue

            info = {
                "especie_bd": especie_codigo,
                "archivo_bd": ruta_wav.name,
                "ruta_relativa_bd": str(ruta_wav.relative_to(RUTA_BD_WAV)),
                "tiempo_inicio": i * SOLAPAMIENTO_SEG,
                "tiempo_fin": i * SOLAPAMIENTO_SEG + DURACION_SEGMENTO_SEG,
            }

            nombres = [f"mfcc_mean_{j+1}" for j in range(NUM_MFCC)] + \
                      [f"mfcc_sd_{j+1}" for j in range(NUM_MFCC)]

            filas.append({**info, **dict(zip(nombres, vec))})

        return pd.DataFrame(filas) if filas else None

    except Exception as e:
        print(f"[ERROR] {ruta_wav.name}: {e}")
        return None


# ============================================================
# FASE 1 BD
# ============================================================

def correr_fase1_bd():
    print("\n=== FASE 1 BD — Extracción de MFCC de la Base de Datos ===")

    carpetas_especie = sorted([p for p in RUTA_BD_WAV.iterdir() if p.is_dir()],
                              key=lambda p: p.name)

    if not carpetas_especie:
        print("⚠️ No se encontraron carpetas de especie en BD_anfibios_wav.")
        return

    print("\nCarpetas detectadas:")
    for c in carpetas_especie:
        print(" -", c.name)

    dfs = []

    for carpeta in carpetas_especie:
        carpeta_nombre = carpeta.name
        archivos = sorted(carpeta.rglob("*.wav"))

        if not archivos:
            print(f"⚠️ {carpeta_nombre}: sin archivos WAV.")
            continue

        print(f"\n🎧 Procesando carpeta {carpeta_nombre} ({len(archivos)} archivos)")

        for wav in tqdm(archivos, desc=carpeta_nombre, unit="arch"):

            # --- CASO ESPECIAL: Cantos Paraguay ---
            if carpeta_nombre == "Cantos Paraguay":
                # Formato esperado: "XX-YYY - Nombre de la especie.wav"
                nombre = wav.stem
                if " - " in nombre:
                    especie_raw = nombre.split(" - ", 1)[1].strip()
                else:
                    especie_raw = "Especie_desconocida_CantosParaguay"
            else:
                # Caso normal: la carpeta es la especie
                especie_raw = carpeta_nombre

            # Normalizar taxonomía
            especie_norm = normalizar_nombre_especie(especie_raw)
            # Convertir a código tipo "Sci_fuscovarius"
            especie_codigo = especie_a_codigo(especie_norm)

            df = procesar_archivo_bd(wav, especie_codigo)
            if df is not None:
                dfs.append(df)

    if not dfs:
        print("⚠️ No se generaron MFCC para la BD.")
        return

    df_bd = pd.concat(dfs, ignore_index=True)
    df_bd.to_parquet(RUTA_FEATURES_BD, index=False)

    print(f"\n💾 MFCC BD guardados en: {RUTA_FEATURES_BD}")
    print(f"   Total segmentos BD: {len(df_bd)}")

    resumen = df_bd.groupby("especie_bd").size().reset_index(name="segmentos")
    resumen.to_csv(RUTA_SALIDA_BD / "resumen_segmentos_por_especie_BD.csv", index=False)

    print("\n📊 Resumen por especie (códigos):")
    print(resumen)


if __name__ == "__main__":
    correr_fase1_bd()
