# =============================================================================
# Script_F_Verificacion_Subclustering.py
# Campaña Nov-Dic 2024 — Paisajes Sonoros Tapytá
# Figura F: Verificacion del subclustering con espectrogramas
#
# Pregunta a responder: "Los 7 subclusters del Cluster 0 realmente separan
# sonidos acusticamente distintos, o es solo matematica?"
#
# Metodo: para cada subcluster (0..6), carga 3 segmentos:
#   - CENTROIDE: el mas cercano al centro del subcluster en el subespacio UMAP
#   - ALEATORIO: uno al azar (para verificar consistencia interna)
#   - EXTREMO:   el mas lejano del centro (periferia del subcluster)
# Si los 3 segmentos de un subcluster suenan/se ven parecidos entre si,
# y distintos entre subclusters → el subclustering es valido.
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import warnings
warnings.filterwarnings('ignore')

# ── Rutas ─────────────────────────────────────────────────────────────────────
SEAGATE_BASE  = r'E:\Campaña diciembre 2024'
SUB_CSV       = (r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local'
                 r'\resultados_HDD_Seagate\Campaña diciembre 2024\PA-17Tapyta'
                 r'\PA-17Tapyta_v2_horario18a06_insectos6a12_cluster0_subclustering_5B.csv')
OUT_DIR       = (r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local'
                 r'\Campaña_Diciembre_2024\05_Figuras_Congreso_SOLABIMA2026\FigF_Verificacion_Subclustering')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Parámetros de audio ────────────────────────────────────────────────────────
SR_TARGET   = 22050
N_FFT       = 1024
HOP_LENGTH  = 256
FMAX        = 8000
SEGMENT_SEC = 4.0

# Colores para los 7 subclusters (mismos que FigC)
SUBCLUSTER_COLORS = {
    0: '#1565c0',  # azul
    1: '#2e7d32',  # verde
    2: '#e65100',  # naranja
    3: '#6a1b9a',  # violeta
    4: '#c62828',  # rojo
    5: '#00695c',  # teal
    6: '#f9a825',  # amarillo oscuro
}

np.random.seed(7)

# ── Cargar CSV de subclustering ───────────────────────────────────────────────
print("Cargando CSV de subclustering del Cluster 0...")
df = pd.read_csv(SUB_CSV)
valid_ids = sorted([s for s in df['subcluster_id'].unique() if s >= 0])
print(f"  Subclusters validos: {valid_ids}")
print(f"  Distribucion:\n{df['subcluster_id'].value_counts().sort_index()}")

# ── Funcion: cargar audio ─────────────────────────────────────────────────────
def load_audio(archivo_origen, t_start):
    parts = archivo_origen.replace('\\', '/').split('/')
    wav_path = os.path.join(SEAGATE_BASE, parts[0], 'Data', parts[-1])
    if not os.path.exists(wav_path):
        return None
    try:
        y, _ = librosa.load(wav_path, sr=SR_TARGET, offset=float(t_start),
                            duration=SEGMENT_SEC, mono=True)
        return y
    except Exception:
        return None

# ── Funcion: espectrograma en dB ──────────────────────────────────────────────
def compute_spec(y):
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    freqs = librosa.fft_frequencies(sr=SR_TARGET, n_fft=N_FFT)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]),
                                   sr=SR_TARGET, hop_length=HOP_LENGTH)
    mask = freqs <= FMAX
    return S_db[mask, :], freqs[mask], times

# ── Seleccionar segmentos para cada subcluster ────────────────────────────────
print("\nSeleccionando segmentos (centroide / aleatorio / extremo)...")

subcluster_segments = {}   # {sid: {'centroide': row, 'aleatorio': row, 'extremo': row}}

for sid in valid_ids:
    sub = df[df['subcluster_id'] == sid].copy()
    # Centroide en el subespacio 2D
    cx = sub['subU1'].mean()
    cy = sub['subU2'].mean()
    sub['dist_centro'] = np.sqrt((sub['subU1'] - cx)**2 + (sub['subU2'] - cy)**2)

    row_centroide = sub.nsmallest(1, 'dist_centro').iloc[0]
    row_extremo   = sub.nlargest(1, 'dist_centro').iloc[0]
    # Aleatorio: excluir centroide y extremo para que sea genuinamente al azar
    mid_pool = sub[(sub.index != row_centroide.name) &
                   (sub.index != row_extremo.name)]
    row_aleatorio = mid_pool.sample(1).iloc[0] if len(mid_pool) > 0 else row_centroide

    subcluster_segments[sid] = {
        'centroide': row_centroide,
        'aleatorio': row_aleatorio,
        'extremo':   row_extremo,
    }
    print(f"  Subcluster {sid}: n={len(sub)}, "
          f"dist_centro_max={sub['dist_centro'].max():.2f}")

# ── Cargar audios y calcular espectrogramas ───────────────────────────────────
print("\nCargando audios del Seagate...")
spec_data = {}   # {sid: {tipo: (S_db, freqs, times) | None}}

TIPOS = ['centroide', 'aleatorio', 'extremo']

for sid in valid_ids:
    spec_data[sid] = {}
    for tipo in TIPOS:
        row = subcluster_segments[sid][tipo]
        y   = load_audio(row['archivo_origen'], row['tiempo_inicio'])
        if y is not None and len(y) > N_FFT:
            S_db, freqs, times = compute_spec(y)
            spec_data[sid][tipo] = (S_db, freqs, times)
        else:
            spec_data[sid][tipo] = None
            print(f"  ⚠ No se pudo cargar: subcluster {sid} / {tipo}")

# ── Construir figura ───────────────────────────────────────────────────────────
print("\nGenerando figura de verificacion...")

n_rows = len(valid_ids)   # 7 subclusters
n_cols = 3                # centroide | aleatorio | extremo

fig_h = 2.2 * n_rows + 1.2
fig = plt.figure(figsize=(14, fig_h))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                       hspace=0.55, wspace=0.25,
                       top=0.94, bottom=0.04,
                       left=0.08, right=0.97)

col_labels = ['Centroide\n(mas cerca del centro)', 'Aleatorio\n(muestra interna)',
              'Extremo\n(periferia del subcluster)']

for col_i, lbl in enumerate(col_labels):
    fig.text(0.08 + col_i * 0.30, 0.965, lbl,
             ha='left', va='top', fontsize=9, fontweight='bold', color='#444444')

for row_i, sid in enumerate(valid_ids):
    color = SUBCLUSTER_COLORS.get(sid, '#555555')

    for col_i, tipo in enumerate(TIPOS):
        ax = fig.add_subplot(gs[row_i, col_i])
        dat = spec_data[sid][tipo]

        if dat is None:
            ax.text(0.5, 0.5, 'No disponible', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='gray')
            ax.set_xticks([]); ax.set_yticks([])
        else:
            S_db, freqs, times = dat
            ax.pcolormesh(times, freqs / 1000, S_db,
                          shading='auto', cmap='magma',
                          vmin=-60, vmax=0, rasterized=True)
            ax.set_ylim(0, FMAX / 1000)
            ax.set_xlim(0, SEGMENT_SEC)
            ax.tick_params(labelsize=7)

            if col_i == 0:
                ax.set_ylabel('kHz', fontsize=8, labelpad=2)
            else:
                ax.set_yticklabels([])

            if row_i == n_rows - 1:
                ax.set_xlabel('Tiempo (s)', fontsize=8, labelpad=2)
            else:
                ax.set_xticklabels([])

        # Borde de color del subcluster
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color(color)

        # Etiqueta de subcluster en la primera columna
        if col_i == 0:
            n_sub = len(df[df['subcluster_id'] == sid])
            ax.set_title(f'Sub-{sid}  (n={n_sub})',
                         fontsize=8.5, fontweight='bold',
                         color=color, pad=3, loc='left')

# ── Titulo general ─────────────────────────────────────────────────────────────
fig.suptitle(
    'Verificacion del subclustering — Cluster 0 de PA-17 (Pastizal)\n'
    'Cada fila = 1 subcluster | Columnas: segmento central / aleatorio / extremo',
    fontsize=11, y=0.995, color='#222222'
)

# ── Texto explicativo ──────────────────────────────────────────────────────────
fig.text(0.5, 0.013,
         'Si los 3 espectrogramas de cada fila son similares entre si (y distintos entre filas), '
         'el subclustering es acusticamente valido.',
         ha='center', fontsize=8, color='#555555', style='italic')

# ── Guardar ────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'FigF_Verificacion_Subclustering_7subclusters.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nFigura guardada: {out_path}")
print("Script F completado.")
