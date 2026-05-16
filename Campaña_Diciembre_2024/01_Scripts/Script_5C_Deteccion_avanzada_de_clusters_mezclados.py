# ============================================================
# Script 5C — Detección avanzada de clusters mezclados
# ------------------------------------------------------------
# Usa:
#   - archivo_origen (ruta relativa del WAV completo)
#   - tiempo_inicio / tiempo_fin
#   - misma atenuación de insectos que Fase 1
# Para:
#   - reconstruir segmentos filtrados
#   - comparar centro vs extremos
#   - decidir si un cluster está mezclado
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import librosa
import soundfile as sf

# ----------------- PARÁMETROS --------------------

NOMBRE_CAMPANIA = "Campaña diciembre 2024"
SUFIJO_CORRIDA = "v2_horario18a06_insectos6a12"

# Ruta del disco externo donde están los WAV completos
DISCO_EXTERNO = Path(r"D:")
 
BASE_RESULTADOS = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local") \
                  / "resultados_HDD_Seagate" / NOMBRE_CAMPANIA

# Umbrales
UMBRAL_TAMANO_MINIMO = 50
UMBRAL_TAMANO_GRANDE = 400
UMBRAL_SIMILITUD = 0.75

N_SEGMENTOS_CENTRO = 16
N_SEGMENTOS_EXTREMOS = 16

SR_TARGET = 22050
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 1024

# ----------------- FUNCIÓN DE ATENUACIÓN (MISMA QUE FASE 1) --------------------

def adaptive_attenuation(y, sr,
                         fmin=4000, fmax=10000,
                         base_gain_db=-9.0, max_extra_db=-6.0,
                         n_fft=2048, hop_length=512, smooth=0.9):
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

# ----------------- RECONSTRUCCIÓN DE SEGMENTOS --------------------

def cargar_segmento_filtrado(fila):
    """
    Carga el WAV completo desde el disco externo,
    aplica atenuación de insectos,
    y devuelve el segmento filtrado entre tiempo_inicio y tiempo_fin.
    """

    archivo_rel = fila["archivo_origen"]  # ej: "BO-31Tapyta/BO-31TAPYTA_20241112_101004.wav"
    sitio = fila["sitio"]

    nombre_wav = Path(archivo_rel).name

    ruta_wav = DISCO_EXTERNO / NOMBRE_CAMPANIA / sitio / "Data" / nombre_wav

    if not ruta_wav.exists():
        print(f"⚠️ WAV no encontrado: {ruta_wav}")
        return None

    try:
        y, sr = sf.read(ruta_wav)
        if y.ndim > 1:
            y = y.mean(axis=1)

        if sr != SR_TARGET:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR_TARGET)
            sr = SR_TARGET

        # Filtrar insectos (igual que Fase 1)
        y_filt = adaptive_attenuation(y, sr)

        # Cortar segmento
        start = int(fila["tiempo_inicio"] * sr)
        end = int(fila["tiempo_fin"] * sr)
        segmento = y_filt[start:end]

        return segmento, sr

    except Exception as e:
        print(f"⚠️ Error cargando segmento: {e}")
        return None

# ----------------- ESPECTRO Y SIMILITUD --------------------

def vector_logmel(segmento, sr):
    S = librosa.feature.melspectrogram(
        y=segmento, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    v = S_db.mean(axis=1)
    v = v - v.mean()
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else None

def similitud_coseno(v1, v2):
    return float(np.dot(v1, v2)) if v1 is not None and v2 is not None else np.nan

# ----------------- ANÁLISIS DE CLUSTER --------------------

def analizar_cluster(df_cluster):
    n = len(df_cluster)
    if n < UMBRAL_TAMANO_MINIMO:
        return n, np.nan, np.nan, np.nan, False

    # Dispersión en UMAP
    dim1 = df_cluster["DIM1"].values
    dim2 = df_cluster["DIM2"].values
    centro = np.array([dim1.mean(), dim2.mean()])
    coords = np.stack([dim1, dim2], axis=1)
    distancias = np.linalg.norm(coords - centro, axis=1)

    radio_medio = distancias.mean()
    radio_maximo = distancias.max()

    # Ordenar por distancia
    df_cluster = df_cluster.copy()
    df_cluster["dist"] = distancias
    df_ord = df_cluster.sort_values("dist")

    # Seleccionar segmentos
    df_centro = df_ord.head(min(N_SEGMENTOS_CENTRO, n))
    df_extremos = df_ord.tail(min(N_SEGMENTOS_EXTREMOS, n))

    # Espectros promedio
    vectores_centro = []
    vectores_extremos = []

    for _, fila in df_centro.iterrows():
        seg = cargar_segmento_filtrado(fila)
        if seg:
            v = vector_logmel(*seg)
            if v is not None:
                vectores_centro.append(v)

    for _, fila in df_extremos.iterrows():
        seg = cargar_segmento_filtrado(fila)
        if seg:
            v = vector_logmel(*seg)
            if v is not None:
                vectores_extremos.append(v)

    if len(vectores_centro) == 0 or len(vectores_extremos) == 0:
        similitud = np.nan
    else:
        v_centro = np.mean(vectores_centro, axis=0)
        v_extremos = np.mean(vectores_extremos, axis=0)
        similitud = similitud_coseno(v_centro, v_extremos)

    # Decisión
    es_grande = n >= UMBRAL_TAMANO_GRANDE
    es_mezclado = es_grande and (not np.isnan(similitud)) and (similitud <= UMBRAL_SIMILITUD)

    return n, radio_medio, radio_maximo, similitud, es_mezclado

# ----------------- PROCESAR SITIO --------------------

def procesar_sitio(sitio):
    print(f"\n=== Analizando sitio: {sitio} ===")

    ruta_f2 = BASE_RESULTADOS / sitio / f"{sitio}_{SUFIJO_CORRIDA}_fase2_umap_hdbscan.csv"
    if not ruta_f2.exists():
        print(f"⚠️ No existe Fase 2 para {sitio}")
        return

    df = pd.read_csv(ruta_f2)

    resultados = []

    for cluster_id, df_c in df.groupby("cluster_hdbscan"):
        print(f" → Cluster {cluster_id} (n={len(df_c)})")
        n, r_med, r_max, sim, mez = analizar_cluster(df_c)
        resultados.append({
            "cluster_hdbscan": cluster_id,
            "n_segmentos": n,
            "radio_medio": r_med,
            "radio_maximo": r_max,
            "similitud_centro_extremos": sim,
            "es_mezclado_recomendado": mez
        })

    df_out = pd.DataFrame(resultados)
    ruta_out = BASE_RESULTADOS / sitio / f"{sitio}_{SUFIJO_CORRIDA}_metricas_5C.csv"
    df_out.to_csv(ruta_out, index=False)
    print(f"   💾 Guardado: {ruta_out}")

# ----------------- MAIN --------------------

def main():
    sitios = [p.name for p in BASE_RESULTADOS.iterdir() if p.is_dir()]
    for sitio in sorted(sitios):
        procesar_sitio(sitio)

if __name__ == "__main__":
    main()
