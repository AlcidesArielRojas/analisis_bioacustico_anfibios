"""
Generación de una grilla comparativa de espectrogramas
------------------------------------------------------

Este script carga tres audios de referencia (formato .mp3) correspondientes
a tres especies del género *Dendropsophus*: D. nanus, D. minutus y D. sanborni.
Cada audio se convierte a un espectrograma de alta calidad y se organiza en
una figura tipo grilla (3 filas × 1 columna), permitiendo comparar visualmente
las diferencias acústicas entre especies.

Pasos:
1. Cargar los archivos .mp3 desde la carpeta indicada.
2. Convertir cada señal a mono y normalizarla.
3. Calcular un espectrograma log-mel o STFT (a elección; aquí STFT).
4. Dibujar cada espectrograma en una fila independiente con títulos claros.
5. Guardar la figura final como .png.

Requisitos:
- librosa
- matplotlib
- numpy
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Ruta base donde están los audios
BASE_DIR = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local\Dendrosophus spp")

# Archivos esperados
ARCHIVOS = {
    "Dendropsophus nanus": "Dendropsophus-nanus_EKrauczuk_Corrientes_AR.mp3",
    "Dendropsophus minutus": "Dendropsophus_minutus_DBuenoPy.mp3",
    "Dendropsophus sanborni": "Dendropsophus_sanborni_DBuenoPy.mp3"
}

# Parámetros de espectrograma
SR = 22050
N_FFT = 1024
HOP = 256

def cargar_audio(ruta):
    y, sr = librosa.load(ruta, sr=SR, mono=True)
    y = y / np.max(np.abs(y))  # normalización
    return y, sr

def generar_espectrograma(y, sr):
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db

def main():
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for ax, (especie, archivo) in zip(axes, ARCHIVOS.items()):
        ruta = BASE_DIR / archivo
        y, sr = cargar_audio(ruta)
        S_db = generar_espectrograma(y, sr)

        img = librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=HOP,
            x_axis="time",
            y_axis="hz",
            cmap="magma",
            ax=ax
        )

        ax.set_title(especie, fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("Frecuencia (Hz)")

    fig.colorbar(img, ax=axes, format="%+2.f dB")
    fig.suptitle("Comparación de espectrogramas de tres especies de Dendropsophus", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    salida = BASE_DIR / "comparacion_espectrogramas.png"
    plt.savefig(salida, dpi=300)
    plt.close()
    print(f"Figura guardada en: {salida}")

if __name__ == "__main__":
    main()