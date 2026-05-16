# =============================================================================
# Script_G_Grillas_Consistencia_Clusters.py
# Campaña Nov-Dic 2024 — Paisajes Sonoros Tapytá
# Figura G: Grillas de consistencia para clusters seleccionados
#
# Para cada cluster "importante" muestra 5 espectrogramas:
#   col 1 → CENTROIDE (mas cercano al centroide UMAP)
#   col 2-4 → ALEATORIOS (3 segmentos al azar del cluster)
#   col 5 → EXTREMO (mas lejano del centroide = periferia del cluster)
#
# Criterio de seleccion de clusters importantes:
#   - PA-17Tapyta: los 4 arquetipos (2=anfibio, 17=insectos, 11=lluvia, 0=mixto)
#   - EU-16Tapyta: ambos clusters (solo hay 2, muy distintos entre si)
#   - BO-31Tapyta: cluster 1 (mas grande, n=3851) y cluster 2 (mas puro, sim=0.831)
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
BASE_RESULTS  = r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\Campaña_Diciembre_2024\02_Resultados_por_sitio'
SEAGATE_BASE  = r'E:\Campaña diciembre 2024'
OUT_DIR       = (r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local'
                 r'\Campaña_Diciembre_2024\05_Figuras_Congreso_SOLABIMA2026\FigG_Consistencia_Clusters')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Clusters a analizar ───────────────────────────────────────────────────────
# Formato: (site_id, cluster_id, etiqueta_descriptiva, color_habitat)
SELECTED_CLUSTERS = [
    # PA-17 — 4 arquetipos
    ('PA-17Tapyta', 2,  'PA-17 | Cl.2 — Vocaliz. anfibio\n(llamados tonales 1-2 kHz)',  '#1565c0'),
    ('PA-17Tapyta', 17, 'PA-17 | Cl.17 — Coro de insectos\n(estridulacion broadband)',  '#2e7d32'),
    ('PA-17Tapyta', 11, 'PA-17 | Cl.11 — Ruido abiotico\n(lluvia — energia difusa)',    '#37474f'),
    ('PA-17Tapyta', 0,  'PA-17 | Cl.0 — Sonido mixto\n(candidato a subclustering)',     '#6a1b9a'),
    # EU-16 — solo 2 clusters, muy contraste entre si
    ('EU-16Tapyta', 0,  'EU-16 | Cl.0 — Eucaliptal tipo A\n(n=4582, Sil alta)',         '#6a1b9a'),
    ('EU-16Tapyta', 1,  'EU-16 | Cl.1 — Eucaliptal tipo B\n(n=95418, cluster masivo)',  '#9c4dcc'),
    # BO-31 — bosque representativo
    ('BO-31Tapyta', 1,  'BO-31 | Cl.1 — Bosque dominante\n(n=3851, sonido principal)',  '#2e7d32'),
    ('BO-31Tapyta', 2,  'BO-31 | Cl.2 — Bosque secundario\n(n=811, muy compacto)',      '#43a047'),
]

# ── Parámetros de audio ────────────────────────────────────────────────────────
SR_TARGET   = 22050
N_FFT       = 1024
HOP_LENGTH  = 256
FMAX        = 8000
SEGMENT_SEC = 4.0
N_RANDOM    = 3       # segmentos aleatorios por cluster

np.random.seed(13)

# ── Cache de CSVs cargados ────────────────────────────────────────────────────
_csv_cache = {}

def get_site_df(site_id):
    if site_id not in _csv_cache:
        site_dir = os.path.join(BASE_RESULTS, site_id)
        umap_csv = [f for f in os.listdir(site_dir)
                    if 'fase2_umap_hdbscan.csv' in f and 'metrics' not in f][0]
        _csv_cache[site_id] = pd.read_csv(os.path.join(site_dir, umap_csv))
    return _csv_cache[site_id]

# ── Funciones utiles ──────────────────────────────────────────────────────────
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

def compute_spec(y):
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    freqs = librosa.fft_frequencies(sr=SR_TARGET, n_fft=N_FFT)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]),
                                   sr=SR_TARGET, hop_length=HOP_LENGTH)
    mask = freqs <= FMAX
    return S_db[mask, :], freqs[mask], times

def select_segments(df_cluster):
    """Retorna (centroide, [3 aleatorios], extremo) como rows del DataFrame."""
    cx = df_cluster['DIM1'].mean()
    cy = df_cluster['DIM2'].mean()
    df_cluster = df_cluster.copy()
    df_cluster['dist'] = np.sqrt((df_cluster['DIM1'] - cx)**2 +
                                  (df_cluster['DIM2'] - cy)**2)
    centroide = df_cluster.nsmallest(1, 'dist').iloc[0]
    extremo   = df_cluster.nlargest(1, 'dist').iloc[0]
    # Pool para aleatorios: excluir centroide y extremo
    pool = df_cluster[(df_cluster.index != centroide.name) &
                      (df_cluster.index != extremo.name)]
    n_rand = min(N_RANDOM, len(pool))
    aleatorios = pool.sample(n=n_rand).itertuples(index=False) if n_rand > 0 else []
    return centroide, list(aleatorios), extremo

# ── Procesar cada cluster seleccionado ───────────────────────────────────────
print("Procesando clusters seleccionados...")
cluster_data = []   # lista de dicts con espectrogramas

for (site_id, cid, label, color) in SELECTED_CLUSTERS:
    print(f"\n  {label[:40]}...")
    df_site = get_site_df(site_id)
    df_cl   = df_site[df_site['cluster_hdbscan'] == cid]
    n_seg   = len(df_cl)
    print(f"    {n_seg} segmentos en el cluster")

    if n_seg < 5:
        print(f"    ⚠ Muy pocos segmentos, omitiendo")
        continue

    centroide, aleatorios, extremo = select_segments(df_cl)

    # Cargar espectrogramas
    specs = []
    rows_ordered = [centroide] + aleatorios + [extremo]
    col_types    = ['Centroide'] + [f'Aleatorio {i+1}' for i in range(len(aleatorios))] + ['Extremo']

    for row in rows_ordered:
        if hasattr(row, 'archivo_origen'):
            y = load_audio(row.archivo_origen, row.tiempo_inicio)
        else:
            y = load_audio(row['archivo_origen'], row['tiempo_inicio'])
        if y is not None and len(y) > N_FFT:
            specs.append(compute_spec(y))
        else:
            specs.append(None)

    cluster_data.append({
        'label':     label,
        'color':     color,
        'n_seg':     n_seg,
        'specs':     specs,
        'col_types': col_types,
        'site':      site_id,
        'cluster':   cid,
    })
    ok = sum(1 for s in specs if s is not None)
    print(f"    Espectrogramas cargados: {ok}/{len(specs)}")

# ── Construir figura ───────────────────────────────────────────────────────────
print("\nGenerando figura de consistencia...")

n_rows = len(cluster_data)
n_cols = 1 + N_RANDOM + 1   # centroide + N_RANDOM + extremo = 5

fig_h = 2.5 * n_rows + 1.5
fig = plt.figure(figsize=(16, fig_h))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                       hspace=0.50, wspace=0.15,
                       top=0.95, bottom=0.04,
                       left=0.15, right=0.97)

# Encabezados de columnas
col_header_labels = ['Centroide\n(cerca del centro)',
                     'Aleatorio 1', 'Aleatorio 2', 'Aleatorio 3',
                     'Extremo\n(periferia)']
col_x_positions = [0.15 + c * (0.82 / n_cols) + 0.082 / n_cols
                   for c in range(n_cols)]
for c, (hdr, xpos) in enumerate(zip(col_header_labels, col_x_positions)):
    style = 'italic' if 'Aleatorio' in hdr else 'normal'
    fw    = 'bold' if c in (0, 4) else 'normal'
    fig.text(xpos, 0.965, hdr, ha='center', va='top',
             fontsize=8, fontweight=fw, fontstyle=style, color='#333333')

for row_i, cd in enumerate(cluster_data):
    color = cd['color']

    for col_i, (spec, ctype) in enumerate(zip(cd['specs'], cd['col_types'])):
        ax = fig.add_subplot(gs[row_i, col_i])

        if spec is None:
            ax.text(0.5, 0.5, 'N/D', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7, color='gray')
            ax.set_xticks([]); ax.set_yticks([])
        else:
            S_db, freqs, times = spec
            ax.pcolormesh(times, freqs / 1000, S_db,
                          shading='auto', cmap='magma',
                          vmin=-60, vmax=0, rasterized=True)
            ax.set_ylim(0, FMAX / 1000)
            ax.set_xlim(0, SEGMENT_SEC)
            ax.tick_params(labelsize=6.5)
            ax.set_yticks([0, 2, 4, 6, 8])

            if col_i > 0:
                ax.set_yticklabels([])
            else:
                ax.set_yticklabels(['0', '2', '4', '6', '8'])
                ax.set_ylabel('kHz', fontsize=7.5, labelpad=2)

            if row_i < n_rows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('s', fontsize=7.5, labelpad=2)

        # Borde
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
            spine.set_color(color)
            # Borde mas grueso en centroide y extremo
            if col_i in (0, n_cols - 1):
                spine.set_linewidth(2.5)

    # Etiqueta de la fila (izquierda)
    fig.text(0.01, 0.95 - row_i * (0.91 / n_rows),
             cd['label'],
             va='top', ha='left', fontsize=8, fontweight='bold',
             color=color)
    fig.text(0.01, 0.95 - row_i * (0.91 / n_rows) - 0.025,
             f'n = {cd["n_seg"]:,} segs.',
             va='top', ha='left', fontsize=7, color='gray')

# ── Titulo ──────────────────────────────────────────────────────────────────────
fig.suptitle(
    'Consistencia interna de clusters acusticos: centroide / aleatorios / extremo\n'
    'PA-17 (Pastizal) · EU-16 (Eucaliptal) · BO-31 (Bosque) | Tapyta, Paraguay',
    fontsize=11, y=0.99, color='#222222'
)

# ── Guardar ────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'FigG_Consistencia_Clusters_CentroideAleatorioExtremo.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nFigura guardada: {out_path}")
print("Script G completado.")
