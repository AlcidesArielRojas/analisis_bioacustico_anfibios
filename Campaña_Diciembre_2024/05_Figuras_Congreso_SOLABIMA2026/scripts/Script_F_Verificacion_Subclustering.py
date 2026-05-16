# =============================================================================
# Script_F_Verificacion_Subclustering.py
# Campaña Nov-Dic 2024 — Paisajes Sonoros Tapytá
# Figuras F1/F2/F3: Verificación del subclustering del Cluster 0 (PA-17)
#
# Diseño: 3 figuras separadas, una por tipo de segmento.
# Cada figura muestra los 7 subclusters en UNA sola fila horizontal,
# lo que permite comparar los patrones espectrales de un vistazo.
#
#   F1 — CENTROIDES : el segmento más cercano al centro de cada subcluster
#   F2 — ALEATORIOS : un segmento al azar (prueba de consistencia interna)
#   F3 — EXTREMOS   : el segmento más lejano del centro (periferia)
#
# Lógica de validación:
#   Si F1 ≈ F2 ≈ F3 dentro de cada subcluster → el subcluster es homogéneo
#   Si F1 difiere entre subclusters            → el subclustering es válido
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker
import librosa
import warnings
warnings.filterwarnings('ignore')

# ── Rutas ─────────────────────────────────────────────────────────────────────
SEAGATE_BASE = r'E:\Campaña diciembre 2024'
SUB_CSV      = (r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local'
                r'\resultados_HDD_Seagate\Campaña diciembre 2024\PA-17Tapyta'
                r'\PA-17Tapyta_v2_horario18a06_insectos6a12_cluster0_subclustering_5B.csv')
OUT_DIR      = (r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local'
                r'\Campaña_Diciembre_2024\05_Figuras_Congreso_SOLABIMA2026'
                r'\FigF_Verificacion_Subclustering')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Parámetros de audio ────────────────────────────────────────────────────────
SR_TARGET   = 22050
N_FFT       = 1024
HOP_LENGTH  = 256
FMAX        = 8000
SEGMENT_SEC = 4.0

# ── Colores por subcluster (mismos que FigC) ──────────────────────────────────
SUB_COLORS = {
    0: '#1565c0',   # azul
    1: '#2e7d32',   # verde
    2: '#e65100',   # naranja
    3: '#6a1b9a',   # violeta
    4: '#c62828',   # rojo
    5: '#00695c',   # teal
    6: '#f9a825',   # amarillo oscuro
}

# Fondo muy claro para cada subcluster (mismo tono que el color, muy diluido)
SUB_BG = {
    0: '#e3f0fc',
    1: '#e8f5e9',
    2: '#fff3e0',
    3: '#f3e5f5',
    4: '#ffebee',
    5: '#e0f2f1',
    6: '#fffde7',
}

np.random.seed(7)

# ── Cargar CSV de subclustering ───────────────────────────────────────────────
print("Cargando CSV de subclustering...")
df = pd.read_csv(SUB_CSV)
valid_ids = sorted([s for s in df['subcluster_id'].unique() if s >= 0])
counts    = df['subcluster_id'].value_counts().sort_index()
print(f"  Subclusters: {valid_ids}")
print(f"  Distribución:\n{counts}")

# ── Funciones utilitarias ─────────────────────────────────────────────────────
def load_audio(archivo_origen, t_start):
    parts    = archivo_origen.replace('\\', '/').split('/')
    wav_path = os.path.join(SEAGATE_BASE, parts[0], 'Data', parts[-1])
    if not os.path.exists(wav_path):
        return None
    try:
        y, _ = librosa.load(wav_path, sr=SR_TARGET, offset=float(t_start),
                            duration=SEGMENT_SEC, mono=True)
        return y
    except Exception:
        return None

def compute_spec(y):
    D     = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db  = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    freqs = librosa.fft_frequencies(sr=SR_TARGET, n_fft=N_FFT)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]),
                                   sr=SR_TARGET, hop_length=HOP_LENGTH)
    # Excluir DC (0 Hz) y aplicar FMAX; devuelve frecuencias en Hz
    mask  = (freqs > 0) & (freqs <= FMAX)
    return S_db[mask, :], freqs[mask], times

def select_three(df_sub):
    """Devuelve (row_centroide, row_aleatorio, row_extremo) en el sub-espacio."""
    df = df_sub.copy()
    cx = df['subU1'].mean()
    cy = df['subU2'].mean()
    df['dist'] = np.sqrt((df['subU1'] - cx)**2 + (df['subU2'] - cy)**2)
    row_c = df.nsmallest(1, 'dist').iloc[0]
    row_e = df.nlargest(1, 'dist').iloc[0]
    pool  = df[(df.index != row_c.name) & (df.index != row_e.name)]
    row_a = pool.sample(1).iloc[0] if len(pool) > 0 else row_c
    return row_c, row_a, row_e

# ── Seleccionar segmentos y cargar espectrogramas ─────────────────────────────
print("\nSeleccionando segmentos y cargando audio...")

all_specs = {}   # {sid: {'centroide': spec|None, 'aleatorio': spec|None, 'extremo': spec|None}}

for sid in valid_ids:
    sub       = df[df['subcluster_id'] == sid]
    row_c, row_a, row_e = select_three(sub)
    all_specs[sid] = {}
    for tipo, row in [('centroide', row_c), ('aleatorio', row_a), ('extremo', row_e)]:
        y = load_audio(row['archivo_origen'], row['tiempo_inicio'])
        if y is not None and len(y) > N_FFT:
            all_specs[sid][tipo] = compute_spec(y)
        else:
            all_specs[sid][tipo] = None
            print(f"  ⚠ Sin audio: Sub-{sid} / {tipo}")

    n = len(sub)
    d_max = sub.assign(
        dist=lambda x: np.sqrt((x['subU1'] - sub['subU1'].mean())**2 +
                               (x['subU2'] - sub['subU2'].mean())**2)
    )['dist'].max()
    print(f"  Sub-{sid}: n={n:>3}, dist_max={d_max:.2f} → OK")

# ── Función generadora de figura ──────────────────────────────────────────────
def build_figure(tipo, fig_id, titulo_principal, subtitulo, filename):
    """
    Genera una figura con 7 paneles apilados verticalmente,
    uno por subcluster (fila), todos del mismo tipo de segmento.
    """
    n = len(valid_ids)   # 7

    # Figura vertical: un subcluster por fila
    fig = plt.figure(figsize=(11, n * 2.6))
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(n, 1, figure=fig,
                           hspace=0.55,
                           top=0.96, bottom=0.04,
                           left=0.09, right=0.97)

    letras = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)']

    for row_i, sid in enumerate(valid_ids):
        ax    = fig.add_subplot(gs[row_i, 0])
        spec  = all_specs[sid].get(tipo)
        color = SUB_COLORS[sid]
        n_seg = counts[sid]

        if spec is not None:
            S_db, freqs, times = spec
            ax.pcolormesh(times, freqs, S_db,
                          shading='auto', cmap='magma',
                          vmin=-60, vmax=0, rasterized=True)
            ax.set_xlim(0, SEGMENT_SEC)
        else:
            ax.set_facecolor('#f0f0f0')
            ax.text(0.5, 0.5, 'N/D', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='gray')

        # ── Eje Y logarítmico en Hz (como el espectrograma de referencia) ──────
        ax.set_yscale('log')
        ax.set_ylim(30, FMAX)
        ax.tick_params(labelsize=7.5)
        ax.set_yticks([100, 500, 1000, 2000, 4000, 8000])
        ax.set_yticklabels(['100', '500', '1k', '2k', '4k', '8k'])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylabel('Hz', fontsize=8.5, labelpad=3)

        if row_i == n - 1:
            ax.set_xlabel('Tiempo (s)', fontsize=9, labelpad=3)
        else:
            ax.set_xticklabels([])

        # ── Borde fino del color del subcluster ───────────────────────────────
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
            spine.set_color(color)

        # ── Título limpio: letra + subcluster + n ─────────────────────────────
        ax.set_title(
            f'{letras[row_i]}  Sub-cluster {sid}  ·  n = {n_seg}',
            fontsize=9.5, fontweight='bold', color=color,
            pad=4, loc='left'
        )

    # ── Título general ─────────────────────────────────────────────────────────
    fig.suptitle(
        f'{fig_id} — {titulo_principal}\n'
        f'Cluster 0 de PA-17 (Pastizal) → 7 subclusters | Tapytá, Paraguay',
        fontsize=11, y=0.99, color='#222222', fontweight='bold'
    )

    # ── Nota al pie ────────────────────────────────────────────────────────────
    fig.text(0.5, 0.01, subtitulo,
             ha='center', fontsize=8.5, color='#555555', style='italic')

    # ── Guardar ───────────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Guardada: {out_path}")


# ── Generar las 3 figuras ─────────────────────────────────────────────────────
print("\n--- Generando FigF1: CENTROIDES ---")
build_figure(
    tipo='centroide',
    fig_id='FigF1',
    titulo_principal='Centroides — el segmento más representativo de cada sub-cluster',
    subtitulo=(
        'Segmento con la mínima distancia al centro del sub-cluster en el espacio UMAP local. '
        'Son los "ejemplares tipo" que definen cada sub-cluster.'
    ),
    filename='FigF1_Centroides_7subclusters.png'
)

print("\n--- Generando FigF2: ALEATORIOS ---")
build_figure(
    tipo='aleatorio',
    fig_id='FigF2',
    titulo_principal='Aleatorios — verificación de consistencia interna',
    subtitulo=(
        'Segmento elegido al azar dentro de cada sub-cluster. '
        'Si se parece al centroide (FigF1), el sub-cluster es internamente homogéneo.'
    ),
    filename='FigF2_Aleatorios_7subclusters.png'
)

print("\n--- Generando FigF3: EXTREMOS ---")
build_figure(
    tipo='extremo',
    fig_id='FigF3',
    titulo_principal='Extremos — el segmento más lejano del centro (periferia del sub-cluster)',
    subtitulo=(
        'Segmento con la máxima distancia al centro en el espacio UMAP local. '
        'Cuanto más se parezca al centroide (FigF1), más compacto y homogéneo es el sub-cluster.'
    ),
    filename='FigF3_Extremos_7subclusters.png'
)

print("\nScript F completado. 3 figuras generadas en:")
print(f"  {OUT_DIR}")
