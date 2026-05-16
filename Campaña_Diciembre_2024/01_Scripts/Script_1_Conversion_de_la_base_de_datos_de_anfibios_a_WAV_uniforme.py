# ================================================================
# Script 1: Conversión de la base de datos de anfibios a WAV uniforme
# (versión con paso previo usando FFmpeg)
# ------------------------------------------------
# DESCRIPCIÓN BREVE
#
# Este script hace DOS cosas:
#
#   1) PASO PREVIO (FFmpeg):
#        - Recorre toda la base en:
#            C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local\BD anfibios
#        - Para cualquier archivo de audio (mp3, wma, m4a, mp4, etc.)
#          que NO sea .wav, genera un .wav en la MISMA carpeta
#          usando FFmpeg (si ya existe el .wav, lo salta).
#
#   2) PASO PRINCIPAL (librosa + soundfile):
#        - Recorre nuevamente toda la base (ya con muchos más .wav),
#        - Carga los .wav,
#        - Los normaliza a:
#             * mono
#             * 22050 Hz
#             * 16-bit PCM
#        - Guarda los archivos convertidos en una carpeta espejo:
#            C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local\BD_anfibios_wav
#          respetando la misma estructura de subcarpetas.
#
# Resultado: una BD de referencia homogénea en BD_anfibios_wav,
# lista para extraer embeddings en los siguientes scripts.
# ================================================================

from pathlib import Path
import subprocess
import soundfile as sf
import librosa

# ---------------- CONFIGURACIÓN ----------------

BASE_DIR = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local")

RUTA_BD_ORIG = BASE_DIR / "BD anfibios"
RUTA_BD_WAV  = BASE_DIR / "BD_anfibios_wav"
RUTA_BD_WAV.mkdir(parents=True, exist_ok=True)

SR_TARGET = 22050
MONO = True

# Ruta exacta a FFmpeg (la que me diste)
FFMPEG_EXE = r"C:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"

# Extensiones que consideramos "audio" para el paso FFmpeg
AUDIO_EXTS_FFMPEG = {
    ".wav", ".mp3", ".wma", ".mp4", ".m4a", ".flac", ".ogg", ".aac"
}

# Extensiones que vamos a procesar en el paso librosa (solo .wav)
AUDIO_EXTS_LIBROSA = {".wav"}


# ---------------- FUNCIONES: PASO PREVIO (FFmpeg) ----------------

def convertir_a_wav_con_ffmpeg(ruta_in: Path, ruta_out: Path):
    """
    Usa FFmpeg para convertir cualquier formato de audio a WAV
    (mono, 22050 Hz). Si FFmpeg falla, simplemente no se genera el archivo.
    """
    ruta_out.parent.mkdir(parents=True, exist_ok=True)

    comando = [
        FFMPEG_EXE,
        "-y",
        "-i", str(ruta_in),
        "-ac", "1",
        "-ar", str(SR_TARGET),
        str(ruta_out)
    ]

    try:
        res = subprocess.run(
            comando,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if res.returncode == 0:
            print(f"✓ FFmpeg: {ruta_in}  →  {ruta_out}")
        else:
            print(f"⚠️ FFmpeg falló con {ruta_in}")
    except Exception as e:
        print(f"⚠️ Error al llamar a FFmpeg para {ruta_in}: {e}")


def paso_previo_ffmpeg():
    """
    Recorre toda la BD_anfibios y para cada archivo de audio que
    NO sea .wav, genera un .wav paralelo usando FFmpeg.
    Si el .wav ya existe, se salta.
    """
    if not RUTA_BD_ORIG.exists():
        print(f"⚠️ No se encontró la base de datos original: {RUTA_BD_ORIG}")
        return

    print("\n================================================")
    print("PASO PREVIO: Conversión a WAV con FFmpeg (en BD anfibios)")
    print("Carpeta origen :", RUTA_BD_ORIG)
    print("================================================\n")

    archivos_audio = [
        p for p in RUTA_BD_ORIG.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS_FFMPEG
    ]

    print(f"Archivos de audio encontrados (para FFmpeg): {len(archivos_audio)}\n")

    for ruta_in in archivos_audio:
        if ruta_in.suffix.lower() == ".wav":
            continue  # ya es WAV

        ruta_out = ruta_in.with_suffix(".wav")
        if ruta_out.exists():
            print(f"⏩ Ya existe WAV, se salta FFmpeg: {ruta_out}")
            continue

        convertir_a_wav_con_ffmpeg(ruta_in, ruta_out)

    print("\n✔️ Paso previo FFmpeg completado.\n")


# ---------------- FUNCIONES: PASO PRINCIPAL (librosa) ----------------

def convertir_a_wav_uniforme(ruta_in: Path, ruta_out: Path):
    """
    Carga un archivo WAV con librosa,
    lo convierte a mono, 22050 Hz y lo guarda como WAV 16-bit PCM.
    """
    try:
        y, sr = librosa.load(str(ruta_in), sr=SR_TARGET, mono=MONO)
    except Exception as e:
        print(f"⚠️ Error al leer {ruta_in}: {e}")
        return

    ruta_out.parent.mkdir(parents=True, exist_ok=True)

    try:
        sf.write(str(ruta_out), y, SR_TARGET, subtype="PCM_16")
        print(f"✓ Convertido (uniforme): {ruta_in}  →  {ruta_out}")
    except Exception as e:
        print(f"⚠️ Error al escribir {ruta_out}: {e}")


def paso_principal_librosa():
    """
    Recorre toda la BD_anfibios (ya con muchos .wav gracias a FFmpeg),
    y genera la versión uniforme en BD_anfibios_wav.
    """
    print("\n================================================")
    print("PASO PRINCIPAL: Normalización a WAV uniforme (librosa + soundfile)")
    print("Carpeta origen :", RUTA_BD_ORIG)
    print("Carpeta destino:", RUTA_BD_WAV)
    print("================================================\n")

    archivos_audio = [
        p for p in RUTA_BD_ORIG.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS_LIBROSA
    ]

    print(f"Archivos .wav encontrados (para normalizar): {len(archivos_audio)}\n")

    for ruta_in in archivos_audio:
        rel = ruta_in.relative_to(RUTA_BD_ORIG)
        ruta_out = (RUTA_BD_WAV / rel).with_suffix(".wav")

        if ruta_out.exists():
            print(f"⏩ Ya existe en BD_anfibios_wav, se salta: {ruta_out}")
            continue

        convertir_a_wav_uniforme(ruta_in, ruta_out)

    print("\n✔️ Conversión completada. BD de anfibios lista en WAV uniforme.")


# ---------------- MAIN ----------------

def main():
    paso_previo_ffmpeg()
    paso_principal_librosa()


if __name__ == "__main__":
    main()