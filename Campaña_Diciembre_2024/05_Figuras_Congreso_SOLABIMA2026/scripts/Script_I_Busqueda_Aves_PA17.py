# =============================================================================
# Script_I_Busqueda_Aves_PA17.py
# Campaña Nov-Dic 2024 — Paisajes Sonoros Tapytá
#
# Búsqueda de clusters con posibles vocalizaciones de aves en PA-17.
# Identificación manual previa:
#   Cl.2  → Vocalización de anfibio  (confirmado)
#   Cl.39 → Vocalización de ave      (confirmado — ANCLA DE BÚSQUEDA)
#   Cl.38 → Lluvia + tormenta        (confirmado)
#   Cl.17 → Coro de insectos         (confirmado)
#   Cl.0  → Mixto vocal + insectos   (confirmado)
#
# Lógica: los clusters más cercanos a Cl.39 en UMAP 2D comparten
# características espectrales con las aves (mayor modulación de
# frecuencia, componentes tonales más agudos que las ranas).
#
# Salidas:
#   1. FigI_UMAP2D_Busqueda_Aves_PA17.png
#      UMAP 2D con Cl.39 (ave) como ancla y candidatos aves destacados.
#      Compara visualmente la posición de aves vs anfibios en el espacio.
#   2. audio_aves/ (directorio)
#      Centroide + aleatorio por cada candidato ave → verificación auditiva.
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import librosa
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_RESULTS = r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\Campaña_Diciembre_2024\02_Resultados_por_sitio'
SEAGATE_BASE = r'E:\Campaña diciembre 2024'
OUT_DIR      = (r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local'
                r'\Campaña_Diciembre_2024\05_Figuras_Congreso_SOLABIMA2026'
                r'\FigI_Busqueda_Aves')
os.makedirs(OUT_DIR, exist_ok=True)
AUDIO_DIR = os.path.join(OUT_DIR, 'audio_aves')
os.makedirs(AUDIO_DIR, exist_ok=True)

SITE        = 'PA-17Tapyta'
SR_TARGET   = 22050
SEGMENT_SEC = 4.0
N_CAND      = 8      # candidatos ave a explorar

np.random.seed(42)

# ── Conocimiento manual acumulado ─────────────────────────────────────────────
CONFIRMED = {
    2:  {'label': 'Cl.2   Anfibio',              'color': '#1565c0', 'cat': 'Anfibio'},
    39: {'label': 'Cl.39  Ave (ancla)',           'color': '#e53935', 'cat': 'Ave'},
    38: {'label': 'Cl.38  Lluvia + tormenta',    'color': '#546e7a', 'cat': 'Abiótico'},
    17: {'label': 'Cl.17  Coro de insectos',     'color': '#2e7d32', 'cat': 'Insecto'},
    0:  {'label': 'Cl.0   Mixto vocal+insect.',  'color': '#6a1b9a', 'cat': 'Mixto'},
    11: {'label': 'Cl.11  ¿Insectos/abiótico?',  'color': '#78909c', 'cat': 'Incierto'},
}
BIRD_ANCHOR  = 39    # ancla para la búsqueda de aves
FROG_ANCHOR  = 2     # anfibio de referencia
EXCLUDE_FROM_SEARCH = set(CONFIRMED.keys())  # no re-buscar ya conocidos

CAT_COLORS = {
    'Anfibio':  '#1565c0',
    'Ave':      '#e53935',
    'Insecto':  '#2e7d32',
    'Abiótico': '#546e7a',
    'Mixto':    '#6a1b9a',
    'Incierto': '#78909c',
}

# ── Cargar datos ──────────────────────────────────────────────────────────────
print(f"Cargando datos de {SITE}...")
site_dir = os.path.join(BASE_RESULTS, SITE)
umap_csv = [f for f in os.listdir(site_dir)
            if 'fase2_umap_hdbscan.csv' in f and 'metrics' not in f][0]
df = pd.read_csv(os.path.join(site_dir, umap_csv))
print(f"  {len(df):,} segmentos  |  {df['cluster_hdbscan'].nunique()-1} clusters válidos")

cluster_ids = sorted([c for c in df['cluster_hdbscan'].unique() if c >= 0])
counts      = df['cluster_hdbscan'].value_counts()

# ── Centroides UMAP 2D ────────────────────────────────────────────────────────
centroids = {}
for cid in cluster_ids:
    sub = df[df['cluster_hdbscan'] == cid]
    centroids[cid] = (sub['DIM1'].mean(), sub['DIM2'].mean())

# ── Selección de candidatos aves ──────────────────────────────────────────────
cx_bird, cy_bird = centroids[BIRD_ANCHOR]
cx_frog, cy_frog = centroids[FROG_ANCHOR]

candidates = []
for cid in cluster_ids:
    if cid in EXCLUDE_FROM_SEARCH:
        continue
    n = int(counts.get(cid, 0))
    if n < 80:
        continue
    cx, cy     = centroids[cid]
    dist_bird  = np.sqrt((cx - cx_bird)**2 + (cy - cy_bird)**2)
    dist_frog  = np.sqrt((cx - cx_frog)**2 + (cy - cy_frog)**2)
    candidates.append({
        'cid':       cid,
        'n':         n,
        'dist_bird': dist_bird,
        'dist_frog': dist_frog,
        'cx': cx, 'cy': cy,
    })

# Ordenar por distancia al ancla de ave
candidates = sorted(candidates, key=lambda x: x['dist_bird'])[:N_CAND]

print(f"\nCandidatos de ave (top {N_CAND} más cercanos a Cl.{BIRD_ANCHOR}):")
print(f"  {'Cl':>4}  {'n':>6}  {'dist_Ave':>9}  {'dist_Anfibio':>12}")
for c in candidates:
    print(f"  {c['cid']:>4}  {c['n']:>6}  {c['dist_bird']:>9.2f}  {c['dist_frog']:>12.2f}")

CAND_COLORS = ['#ff6f00', '#f57c00', '#ef6c00', '#e65100',
               '#d84315', '#bf360c', '#c62828', '#b71c1c']

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
    d = df_cl.copy()
    d['_d'] = np.sqrt((d['DIM1'] - d['DIM1'].mean())**2 +
                      (d['DIM2'] - d['DIM2'].mean())**2)
    return d.nsmallest(1, '_d').iloc[0]

def random_row(df_cl):
    return df_cl.sample(1).iloc[0]

# ── Exportar audio de los clusters conocidos relevantes ───────────────────────
print("\nExportando audio de anclas (anfibio y ave confirmada)...")
for cid in [FROG_ANCHOR, BIRD_ANCHOR]:
    info = CONFIRMED[cid]
    sub  = df[df['cluster_hdbscan'] == cid]
    for tipo, fn_row in [('centroide', centroid_row), ('aleatorio', random_row)]:
        row = fn_row(sub)
        y   = load_audio(row['archivo_origen'], row['tiempo_inicio'])
        if y is not None:
            safe  = info['label'].replace(' ', '_').replace('.', '').replace('/', '-') \
                                 .replace('(', '').replace(')', '').replace('á','a') \
                                 .replace('ó','o').replace('é','e').replace('+','mas')
            fname = os.path.join(AUDIO_DIR, f'ANCLA_{safe}_{tipo}.wav')
            sf.write(fname, y, SR_TARGET)
            print(f"  ✓ {info['label']} [{tipo}]")

# ── Exportar audio de candidatos aves ─────────────────────────────────────────
print("\nExportando audio de candidatos aves...")
for c in candidates:
    cid = c['cid']
    sub = df[df['cluster_hdbscan'] == cid]
    for tipo, fn_row in [('centroide', centroid_row), ('aleatorio', random_row)]:
        row = fn_row(sub)
        y   = load_audio(row['archivo_origen'], row['tiempo_inicio'])
        if y is not None:
            fname = os.path.join(AUDIO_DIR,
                f'CANDIDATO_AVE_Cl{cid:02d}_n{c["n"]}_{tipo}.wav')
            sf.write(fname, y, SR_TARGET)
            print(f"  ✓ Cl.{cid:>2} n={c['n']:>4} [{tipo}]")
        else:
            print(f"  ✗ Cl.{cid:>2} — sin audio [{tipo}]")

# ── Figura UMAP 2D ────────────────────────────────────────────────────────────
print("\nGenerando figura UMAP 2D — búsqueda de aves...")

fig, axes = plt.subplots(1, 2, figsize=(18, 9),
                          gridspec_kw={'width_ratios': [2, 1], 'wspace': 0.30})
fig.patch.set_facecolor('white')

# ══ Panel izquierdo: UMAP completo de PA-17 ══════════════════════════════════
ax = axes[0]

# Ruido
mask_noise = df['cluster_hdbscan'] == -1
ax.scatter(df.loc[mask_noise, 'DIM1'], df.loc[mask_noise, 'DIM2'],
           c='#f5f5f5', s=0.3, alpha=0.15, rasterized=True, linewidths=0)

# Clusters generales (gris + número)
for cid in cluster_ids:
    if cid in CONFIRMED or any(c['cid'] == cid for c in candidates):
        continue
    mask = df['cluster_hdbscan'] == cid
    ax.scatter(df.loc[mask, 'DIM1'], df.loc[mask, 'DIM2'],
               c='#dddddd', s=0.4, alpha=0.20, rasterized=True, linewidths=0)
    cx, cy = centroids[cid]
    ax.text(cx, cy, str(cid), fontsize=5, ha='center', va='center',
            color='#bbbbbb', fontweight='bold', zorder=2)

# Candidatos aves (naranja/rojo)
handles_cand = []
for i, c in enumerate(candidates):
    cid   = c['cid']
    color = CAND_COLORS[i]
    mask  = df['cluster_hdbscan'] == cid
    ax.scatter(df.loc[mask, 'DIM1'], df.loc[mask, 'DIM2'],
               c=color, s=6, alpha=0.65, rasterized=True, linewidths=0, zorder=4)
    cx, cy = c['cx'], c['cy']
    ax.text(cx, cy + 0.18, f'Cl.{cid}', fontsize=8, ha='center', va='bottom',
            color=color, fontweight='bold', zorder=6)
    handles_cand.append(
        mpatches.Patch(facecolor=color,
                       label=f'Cl.{cid:<3} n={c["n"]:>5,}  d={c["dist_bird"]:.1f}')
    )

# Clusters confirmados
handles_conf = []
annot_offsets = {
    2:  ( 2.5,  1.0),
    39: (-3.5,  1.5),
    38: ( 2.0, -2.0),
    17: (-3.0, -1.5),
    0:  ( 2.5, -1.0),
    11: (-3.0,  1.0),
}
for cid, info in CONFIRMED.items():
    color = info['color']
    mask  = df['cluster_hdbscan'] == cid
    ax.scatter(df.loc[mask, 'DIM1'], df.loc[mask, 'DIM2'],
               c=color, s=8, alpha=0.80, rasterized=True, linewidths=0, zorder=5)
    cx, cy = centroids[cid]
    ox, oy = annot_offsets.get(cid, (2.0, 1.0))
    ax.annotate(
        info['label'],
        xy=(cx, cy), xytext=(cx + ox, cy + oy),
        fontsize=8.5, fontweight='bold', color=color,
        arrowprops=dict(arrowstyle='->', color=color, lw=1.3,
                        connectionstyle='arc3,rad=-0.15'),
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  alpha=0.90, edgecolor=color, linewidth=1.0),
        zorder=8
    )
    handles_conf.append(mpatches.Patch(facecolor=color, label=info['label']))

# Línea que conecta ancla ave con ancla anfibio
ax.plot([cx_frog, cx_bird], [cy_frog, cy_bird],
        '--', color='#888888', lw=1.2, alpha=0.5, zorder=3)
ax.text((cx_frog + cx_bird)/2, (cy_frog + cy_bird)/2 + 0.3,
        f'dist={np.sqrt((cx_bird-cx_frog)**2+(cy_bird-cy_frog)**2):.1f}',
        fontsize=7.5, color='#888888', ha='center', style='italic')

ax.set_xlabel('UMAP Dimensión 1', fontsize=10)
ax.set_ylabel('UMAP Dimensión 2', fontsize=10)
ax.set_title(f'a)  UMAP 2D — PA-17 Tapytá\n'
             f'Ancla ave = Cl.{BIRD_ANCHOR} | Ancla anfibio = Cl.{FROG_ANCHOR}',
             fontsize=10.5, fontweight='bold', pad=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=8.5)

leg1 = ax.legend(handles=handles_conf, loc='upper left',
                 title='Clusters confirmados', title_fontsize=8.5,
                 fontsize=8, frameon=True, framealpha=0.92, edgecolor='#cccccc')
ax.add_artist(leg1)
ax.legend(handles=handles_cand, loc='lower right',
          title=f'Candidatos ave (top {N_CAND})\nd = dist. a Cl.{BIRD_ANCHOR}',
          title_fontsize=8.5,
          fontsize=7.5, frameon=True, framealpha=0.92, edgecolor='#cccccc')

# ══ Panel derecho: tabla resumen de candidatos ════════════════════════════════
ax2 = axes[1]
ax2.axis('off')

# Título del panel
ax2.text(0.5, 0.98, 'b)  Candidatos de ave', transform=ax2.transAxes,
         fontsize=11, fontweight='bold', ha='center', va='top')
ax2.text(0.5, 0.93,
         'Ordenados por similitud acústica\ncon Cl.39 (ave confirmada)',
         transform=ax2.transAxes, fontsize=8.5, ha='center', va='top',
         color='#555555', style='italic')

# Tabla
col_labels = ['Cluster', 'n segs', 'dist Ave', 'dist Anfibio']
table_data = [[f'Cl.{c["cid"]}', f'{c["n"]:,}',
               f'{c["dist_bird"]:.2f}', f'{c["dist_frog"]:.2f}']
              for c in candidates]

y_start = 0.84
row_h   = 0.072
# Header
for j, lbl in enumerate(col_labels):
    xpos = 0.05 + j * 0.24
    ax2.text(xpos, y_start, lbl, transform=ax2.transAxes,
             fontsize=8.5, fontweight='bold', va='top', color='#333333')

ax2.plot([0.02, 0.98], [y_start - 0.015, y_start - 0.015],
         transform=ax2.transAxes, color='#cccccc', lw=1.0, clip_on=False)

# Filas
for i, (row, c) in enumerate(zip(table_data, candidates)):
    y = y_start - row_h * (i + 1)
    color = CAND_COLORS[i]
    # Fondo alternado
    if i % 2 == 0:
        ax2.add_patch(plt.Rectangle(
            (0.02, y - 0.005), 0.96, row_h - 0.005,
            transform=ax2.transAxes, facecolor='#fafafa',
            edgecolor='none', zorder=0, clip_on=False))
    for j, val in enumerate(row):
        xpos = 0.05 + j * 0.24
        weight = 'bold' if j == 0 else 'normal'
        clr    = color if j == 0 else '#333333'
        ax2.text(xpos, y + row_h * 0.35, val,
                 transform=ax2.transAxes,
                 fontsize=8.5, fontweight=weight, va='center', color=clr)

# Nota metodológica
y_nota = y_start - row_h * (N_CAND + 1.5)
ax2.text(0.5, y_nota,
         'Escuchar WAVs en audio_aves/\npara verificar manualmente.',
         transform=ax2.transAxes, fontsize=8, ha='center',
         va='top', color='#666666', style='italic',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff8e1',
                   edgecolor='#f9a825', alpha=0.8))

# Leyenda de categorías confirmadas
y_leg = y_nota - 0.16
ax2.text(0.5, y_leg, 'Categorías acústicas confirmadas:',
         transform=ax2.transAxes, fontsize=8.5, ha='center',
         va='top', fontweight='bold', color='#333333')

cat_info = [
    ('Anfibio',  CAT_COLORS['Anfibio'],  'Llamados tónales 1–3 kHz'),
    ('Ave',      CAT_COLORS['Ave'],      'Vocalizaciones f-moduladas'),
    ('Insecto',  CAT_COLORS['Insecto'],  'Estridulación broadband'),
    ('Abiótico', CAT_COLORS['Abiótico'], 'Lluvia / tormenta'),
    ('Mixto',    CAT_COLORS['Mixto'],    'Superposición de fuentes'),
]
for k, (cat, col, desc) in enumerate(cat_info):
    y_cat = y_leg - 0.065 * (k + 1)
    ax2.add_patch(plt.Rectangle((0.05, y_cat - 0.015), 0.04, 0.04,
                                transform=ax2.transAxes,
                                facecolor=col, edgecolor='none'))
    ax2.text(0.12, y_cat + 0.005, f'{cat}: {desc}',
             transform=ax2.transAxes, fontsize=7.5, va='center', color='#333333')

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    'Búsqueda de clusters de aves en PA-17 (Pastizal) — Tapytá, Paraguay\n'
    'Identificación manual: Cl.39=ave, Cl.2=anfibio, Cl.38=tormenta, Cl.17=insectos, Cl.0=mixto',
    fontsize=11, y=0.99, fontweight='bold', color='#222222'
)

# ── Guardar ───────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'FigI_UMAP2D_Busqueda_Aves_PA17.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n  Figura: {out_path}")
print(f"  Audio : {AUDIO_DIR}")
print(f"\nScript I completado.")
print(f"\nArchivos para escuchar (comparar con ancla de ave Cl.39):")
for c in candidates:
    print(f"  CANDIDATO_AVE_Cl{c['cid']:02d}_n{c['n']}_centroide.wav  "
          f"(dist_ave={c['dist_bird']:.2f})")
