# =============================================================================
# Script_H_Exploracion_Vocalizaciones_PA17.py
# Campaña Nov-Dic 2024 — Paisajes Sonoros Tapytá
#
# Objetivo: Identificar todos los clusters de PA-17 que pueden contener
# vocalizaciones de anfibios (potencialmente otras especies).
#
# Salidas:
#   1. FigH_UMAP2D_Candidatos_Vocalizaciones_PA17.png
#      Mapa UMAP 2D completo de PA-17 con los 4 arquetipos conocidos y
#      los 8 candidatos vocales más cercanos al Cluster 2.
#
#   2. audio_candidatos/ (directorio)
#      Un WAV de centroide + un WAV aleatorio por cada candidato vocal
#      y por cada arquetipo conocido → para verificación auditiva.
#
# Lógica de selección de candidatos:
#   - Se excluyen los 4 arquetipos ya conocidos (0, 2, 11, 17)
#   - Se ordena el resto por distancia euclidiana al centroide del Cl.2
#     en el espacio UMAP 2D (DIM1/DIM2)
#   - Se toman los 8 más cercanos con n >= 80 segmentos
#   Razonamiento: en UMAP, sonidos acústicamente similares se agrupan cerca.
#   Los clusters vecinos al Cl.2 (vocalizacion de anfibio) tienen más
#   probabilidad de también ser vocalizaciones, quizás de otras especies.
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import librosa
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_RESULTS = r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\Campaña_Diciembre_2024\02_Resultados_por_sitio'
SEAGATE_BASE = r'E:\Campaña diciembre 2024'
OUT_DIR      = (r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local'
                r'\Campaña_Diciembre_2024\05_Figuras_Congreso_SOLABIMA2026'
                r'\FigH_Exploracion_Vocalizaciones')
os.makedirs(OUT_DIR, exist_ok=True)
AUDIO_DIR = os.path.join(OUT_DIR, 'audio_candidatos')
os.makedirs(AUDIO_DIR, exist_ok=True)

SITE        = 'PA-17Tapyta'
SR_TARGET   = 22050
SEGMENT_SEC = 4.0
N_CAND      = 8       # número de candidatos vocales a seleccionar

np.random.seed(42)

# ── Cargar datos UMAP de PA-17 ────────────────────────────────────────────────
print(f"Cargando datos de {SITE}...")
site_dir = os.path.join(BASE_RESULTS, SITE)
umap_csv = [f for f in os.listdir(site_dir)
            if 'fase2_umap_hdbscan.csv' in f and 'metrics' not in f][0]
df = pd.read_csv(os.path.join(site_dir, umap_csv))
print(f"  {len(df):,} segmentos totales")

cluster_ids = sorted([c for c in df['cluster_hdbscan'].unique() if c >= 0])
counts      = df['cluster_hdbscan'].value_counts()
print(f"  Clusters válidos: {len(cluster_ids)}  |  Ruido: {(df['cluster_hdbscan']==-1).sum():,} segs")

# ── Centroides en espacio UMAP 2D ─────────────────────────────────────────────
centroids = {}
for cid in cluster_ids:
    sub = df[df['cluster_hdbscan'] == cid]
    centroids[cid] = (sub['DIM1'].mean(), sub['DIM2'].mean())

# ── Arquetipos conocidos ───────────────────────────────────────────────────────
KNOWN = {
    2:  {'label': 'Cl.2  Vocaliz. anfibio',          'color': '#1565c0'},
    17: {'label': 'Cl.17  Coro insectos',             'color': '#2e7d32'},
    11: {'label': 'Cl.11  Ruido abiótico / insectos', 'color': '#455a64'},
    0:  {'label': 'Cl.0  Mixto (vocal + insectos)',   'color': '#6a1b9a'},
}

# ── Selección de candidatos vocales ───────────────────────────────────────────
cx2, cy2 = centroids[2]

candidates = []
for cid in cluster_ids:
    if cid in KNOWN:
        continue
    n = int(counts.get(cid, 0))
    if n < 80:          # muy pequeño: poco material para evaluar
        continue
    cx, cy     = centroids[cid]
    dist_cl2   = np.sqrt((cx - cx2)**2 + (cy - cy2)**2)
    candidates.append({'cid': cid, 'n': n, 'dist': dist_cl2, 'cx': cx, 'cy': cy})

candidates = sorted(candidates, key=lambda x: x['dist'])[:N_CAND]

print(f"\nCandidatos seleccionados (top {N_CAND} más cercanos a Cl.2 en UMAP):")
print(f"  {'Cluster':>7}  {'n':>6}  dist_Cl2")
for c in candidates:
    print(f"  Cl.{c['cid']:>2}      {c['n']:>5}   {c['dist']:.2f}")

# Paleta para candidatos (gradiente naranja-rojo)
CAND_COLORS = ['#e65100', '#f57c00', '#ff8f00', '#ffa000',
               '#d84315', '#bf360c', '#c62828', '#ad1457']

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

def centroid_row(df_cl):
    """Devuelve la fila más cercana al centroide del cluster en UMAP 2D."""
    d = df_cl.copy()
    d['_dist'] = np.sqrt((d['DIM1'] - d['DIM1'].mean())**2 +
                         (d['DIM2'] - d['DIM2'].mean())**2)
    return d.nsmallest(1, '_dist').iloc[0]

def random_row(df_cl):
    return df_cl.sample(1).iloc[0]

# ── Exportar audio: arquetipos conocidos ──────────────────────────────────────
print("\nExportando audio de arquetipos conocidos...")
for cid, info in KNOWN.items():
    sub = df[df['cluster_hdbscan'] == cid]
    for tipo, fn_row in [('centroide', centroid_row), ('aleatorio', random_row)]:
        row = fn_row(sub)
        y   = load_audio(row['archivo_origen'], row['tiempo_inicio'])
        if y is not None:
            safe = info['label'].replace(' ', '_').replace('.', '').replace('/', '-') \
                                .replace('(', '').replace(')', '').replace('á', 'a') \
                                .replace('ó', 'o').replace('é', 'e').replace('í', 'i')
            fname = os.path.join(AUDIO_DIR, f'ARQUETIPO_{safe}_{tipo}.wav')
            sf.write(fname, y, SR_TARGET)
            print(f"  ✓ {info['label']} [{tipo}]")
        else:
            print(f"  ✗ {info['label']} [{tipo}] — sin audio")

# ── Exportar audio: candidatos vocales ────────────────────────────────────────
print("\nExportando audio de candidatos vocales...")
for c in candidates:
    cid = c['cid']
    sub = df[df['cluster_hdbscan'] == cid]
    for tipo, fn_row in [('centroide', centroid_row), ('aleatorio', random_row)]:
        row = fn_row(sub)
        y   = load_audio(row['archivo_origen'], row['tiempo_inicio'])
        if y is not None:
            fname = os.path.join(AUDIO_DIR,
                                 f'CANDIDATO_Cl{cid:02d}_n{c["n"]}_{tipo}.wav')
            sf.write(fname, y, SR_TARGET)
            print(f"  ✓ Cl.{cid:>2} (n={c['n']:>4}) [{tipo}]")
        else:
            print(f"  ✗ Cl.{cid:>2} (n={c['n']:>4}) [{tipo}] — sin audio")

# ── Figura UMAP 2D ────────────────────────────────────────────────────────────
print("\nGenerando figura UMAP 2D de PA-17...")

fig, ax = plt.subplots(figsize=(14, 11))
fig.patch.set_facecolor('white')

# 1. Puntos de ruido HDBSCAN
mask_noise = df['cluster_hdbscan'] == -1
ax.scatter(df.loc[mask_noise, 'DIM1'], df.loc[mask_noise, 'DIM2'],
           c='#f0f0f0', s=0.3, alpha=0.15, rasterized=True, linewidths=0,
           label='_nolegend_')

# 2. Clusters generales (gris) con número de cluster en el centroide
for cid in cluster_ids:
    if cid in KNOWN or any(c['cid'] == cid for c in candidates):
        continue
    mask = df['cluster_hdbscan'] == cid
    ax.scatter(df.loc[mask, 'DIM1'], df.loc[mask, 'DIM2'],
               c='#cccccc', s=0.4, alpha=0.25, rasterized=True,
               linewidths=0, label='_nolegend_')
    cx, cy = centroids[cid]
    ax.text(cx, cy, str(cid), fontsize=5.5, ha='center', va='center',
            color='#aaaaaa', fontweight='bold', zorder=3)

# 3. Candidatos vocales (naranja/rojo, más visibles)
handles_cand = []
for i, c in enumerate(candidates):
    cid   = c['cid']
    color = CAND_COLORS[i]
    mask  = df['cluster_hdbscan'] == cid
    ax.scatter(df.loc[mask, 'DIM1'], df.loc[mask, 'DIM2'],
               c=color, s=5, alpha=0.65, rasterized=True,
               linewidths=0, zorder=4)
    cx, cy = c['cx'], c['cy']
    ax.text(cx, cy + 0.15, f'Cl.{cid}', fontsize=8.5, ha='center', va='bottom',
            color=color, fontweight='bold', zorder=6)
    handles_cand.append(
        mpatches.Patch(facecolor=color,
                       label=f'Cl.{cid:<3}  n={c["n"]:>5,}  d={c["dist"]:.1f}')
    )

# 4. Arquetipos conocidos (grandes y anotados)
handles_known = []
for cid, info in KNOWN.items():
    color = info['color']
    mask  = df['cluster_hdbscan'] == cid
    ax.scatter(df.loc[mask, 'DIM1'], df.loc[mask, 'DIM2'],
               c=color, s=10, alpha=0.80, rasterized=True,
               linewidths=0, zorder=5)
    cx, cy = centroids[cid]
    # Flecha de anotación
    offset = {'x': 2.0, 'y': 1.2}
    ax.annotate(
        info['label'],
        xy=(cx, cy),
        xytext=(cx + offset['x'], cy + offset['y']),
        fontsize=9, fontweight='bold', color=color,
        arrowprops=dict(arrowstyle='->', color=color, lw=1.4,
                        connectionstyle='arc3,rad=-0.1'),
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  alpha=0.90, edgecolor=color, linewidth=1.0),
        zorder=8
    )
    handles_known.append(mpatches.Patch(facecolor=color, label=info['label']))

# ── Decoraciones ──────────────────────────────────────────────────────────────
ax.set_xlabel('UMAP Dimensión 1', fontsize=11)
ax.set_ylabel('UMAP Dimensión 2', fontsize=11)
ax.set_title(
    'Exploración de clusters con posibles vocalizaciones — PA-17 (Pastizal) | Tapytá\n'
    f'Los {N_CAND} candidatos vocales se seleccionaron por proximidad acústica al Cluster 2 '
    f'(eje UMAP 2D, dist. euclidiana)',
    fontsize=11, fontweight='bold', pad=10, color='#222222'
)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=9)

# Nota metodológica
ax.text(0.01, 0.01,
        'Los números grises son clusters no seleccionados.\n'
        'Escuchar los WAV en audio_candidatos/ para verificar.',
        transform=ax.transAxes, fontsize=7.5, color='#777777',
        va='bottom', style='italic')

# Leyendas
leg_known = ax.legend(
    handles=handles_known, loc='upper left',
    title='Arquetipos conocidos', title_fontsize=9,
    fontsize=8.5, frameon=True, framealpha=0.92, edgecolor='#aaaaaa',
    handlelength=1.2
)
ax.add_artist(leg_known)
ax.legend(
    handles=handles_cand, loc='lower right',
    title=f'Candidatos vocales (top {N_CAND})\n[d = dist. a Cl.2]',
    title_fontsize=8.5,
    fontsize=8, frameon=True, framealpha=0.92, edgecolor='#aaaaaa',
    handlelength=1.2
)

# ── Guardar ───────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'FigH_UMAP2D_Candidatos_Vocalizaciones_PA17.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n  Figura: {out_path}")
print(f"  Audio : {AUDIO_DIR}")
print(f"\nScript H completado.")
print(f"\nArchivos de audio generados (escucharlos en orden):")
for c in candidates:
    print(f"  CANDIDATO_Cl{c['cid']:02d}_n{c['n']}_centroide.wav")
    print(f"  CANDIDATO_Cl{c['cid']:02d}_n{c['n']}_aleatorio.wav")
