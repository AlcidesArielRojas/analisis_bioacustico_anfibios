# =============================================================================
# Script_C_Subclustering_AntesDespues.py
# Campaña Nov-Dic 2024 — Paisajes Sonoros Tapytá
# Figura C: Antes/Después del subclustering jerárquico
# Panel izquierdo:  UMAP global PA-17 con Cluster 0 resaltado
# Panel derecho:    Subclustering local del Cluster 0 (7 subclusters)
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_RESULTS  = r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\Campaña_Diciembre_2024\02_Resultados_por_sitio'
BASE_SEAGATE  = r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\resultados_HDD_Seagate\Campaña diciembre 2024'
OUT_DIR       = r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\Campaña_Diciembre_2024\05_Figuras_Congreso_SOLABIMA2026\FigC_Subclustering'
os.makedirs(OUT_DIR, exist_ok=True)

SITE          = 'PA-17Tapyta'
TARGET_CLUSTER = 0          # El cluster mixto que fue subclustered
ACCENT_COLOR  = '#e65100'   # Naranja Pastizal

# ── Cargar datos ──────────────────────────────────────────────────────────────
print("Cargando UMAP global de PA-17...")
site_dir = os.path.join(BASE_RESULTS, SITE)
umap_csv = [f for f in os.listdir(site_dir)
            if 'fase2_umap_hdbscan.csv' in f and 'metrics' not in f][0]
df_global = pd.read_csv(os.path.join(site_dir, umap_csv),
                         usecols=['DIM1', 'DIM2', 'cluster_hdbscan'])
print(f"  {len(df_global):,} segmentos cargados")

print("Cargando datos de subclustering del Cluster 0...")
sub_csv = os.path.join(BASE_SEAGATE, SITE,
    f'{SITE}_v2_horario18a06_insectos6a12_cluster{TARGET_CLUSTER}_subclustering_5B.csv')
df_sub = pd.read_csv(sub_csv, usecols=['subU1', 'subU2', 'subcluster_id'])
print(f"  {len(df_sub):,} puntos en el subespacio")
subclusters = sorted(df_sub['subcluster_id'].unique())
print(f"  Subclusters encontrados: {subclusters}")

# ── Paleta de colores para subclusters ────────────────────────────────────────
SUBCLUSTER_COLORS = [
    '#1565c0',  # 0 azul
    '#2e7d32',  # 1 verde
    '#e65100',  # 2 naranja
    '#6a1b9a',  # 3 violeta
    '#c62828',  # 4 rojo
    '#00695c',  # 5 teal
    '#f9a825',  # 6 amarillo oscuro
]
NOISE_COLOR    = '#cccccc'
GRAY_OTHERS    = '#dddddd'
HIGHLIGHT_CL0  = '#ff6f00'    # naranja brillante para el cluster a resaltar

# ── Construir figura ──────────────────────────────────────────────────────────
print("\nGenerando figura antes/después...")

fig = plt.figure(figsize=(17, 7.5))
fig.patch.set_facecolor('white')
gs = GridSpec(1, 2, figure=fig, wspace=0.28,
              left=0.07, right=0.97, top=0.87, bottom=0.09)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL IZQUIERDO: UMAP global con Cluster 0 resaltado
# ══════════════════════════════════════════════════════════════════════════════
ax_left = fig.add_subplot(gs[0])

# 1. Todos los demás puntos en gris muy claro
mask_others = (df_global['cluster_hdbscan'] != TARGET_CLUSTER) & \
              (df_global['cluster_hdbscan'] >= 0)
mask_noise  = df_global['cluster_hdbscan'] == -1
mask_target = df_global['cluster_hdbscan'] == TARGET_CLUSTER

ax_left.scatter(df_global.loc[mask_noise, 'DIM1'],
                df_global.loc[mask_noise, 'DIM2'],
                c='#f0f0f0', s=0.4, alpha=0.15, rasterized=True, linewidths=0)

ax_left.scatter(df_global.loc[mask_others, 'DIM1'],
                df_global.loc[mask_others, 'DIM2'],
                c=GRAY_OTHERS, s=0.6, alpha=0.35, rasterized=True, linewidths=0)

# 2. Cluster 0 resaltado en naranja brillante
ax_left.scatter(df_global.loc[mask_target, 'DIM1'],
                df_global.loc[mask_target, 'DIM2'],
                c=HIGHLIGHT_CL0, s=2.5, alpha=0.80, rasterized=True,
                linewidths=0, zorder=5)

# Contorno del cluster (convex hull aproximado con elipse)
cl0_x = df_global.loc[mask_target, 'DIM1'].values
cl0_y = df_global.loc[mask_target, 'DIM2'].values
from matplotlib.patches import Ellipse
cx, cy = cl0_x.mean(), cl0_y.mean()
sx, sy = cl0_x.std() * 2.5, cl0_y.std() * 2.5
ell = Ellipse((cx, cy), width=sx*2, height=sy*2,
              edgecolor=HIGHLIGHT_CL0, facecolor='none',
              linewidth=2.0, linestyle='--', zorder=6, alpha=0.85)
ax_left.add_patch(ell)

# Flecha + etiqueta
ax_left.annotate('Cluster 0\n(mixto)',
                 xy=(cx, cy), xytext=(cx + 2.5, cy - 3.5),
                 fontsize=9, fontweight='bold', color=HIGHLIGHT_CL0,
                 arrowprops=dict(arrowstyle='->', color=HIGHLIGHT_CL0,
                                 lw=1.8, connectionstyle='arc3,rad=-0.2'),
                 ha='left')

# Stats box
n_cl0 = mask_target.sum()
ax_left.text(0.02, 0.97,
             f'Cluster 0\nn = {n_cl0:,} segmentos\n'
             f'dist. media = 1.37\n→ candidato a subclustering',
             transform=ax_left.transAxes, fontsize=8, va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff8e1',
                       edgecolor=HIGHLIGHT_CL0, alpha=0.92))

ax_left.set_xlabel('UMAP dim. 1', fontsize=9)
ax_left.set_ylabel('UMAP dim. 2', fontsize=9)
ax_left.set_title('a)  Antes del subclustering\n'
                   'UMAP global PA-17 | Cluster 0 resaltado (naranja)',
                   fontsize=10, fontweight='bold', color='#333333', pad=7)
ax_left.spines['top'].set_visible(False)
ax_left.spines['right'].set_visible(False)
ax_left.tick_params(labelsize=8)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL DERECHO: espacio local del subclustering
# ══════════════════════════════════════════════════════════════════════════════
ax_right = fig.add_subplot(gs[1])

valid_sub_ids = [s for s in subclusters if s >= 0]
noise_mask_s  = df_sub['subcluster_id'] == -1

# Ruido del subespacio
if noise_mask_s.any():
    ax_right.scatter(df_sub.loc[noise_mask_s, 'subU1'],
                     df_sub.loc[noise_mask_s, 'subU2'],
                     c=NOISE_COLOR, s=6, alpha=0.4, rasterized=True,
                     linewidths=0, zorder=1)

# Subclusters con colores distintos
handles_sub = []
for i, sid in enumerate(valid_sub_ids):
    clr = SUBCLUSTER_COLORS[i % len(SUBCLUSTER_COLORS)]
    mask_s = df_sub['subcluster_id'] == sid
    n_s = mask_s.sum()
    ax_right.scatter(df_sub.loc[mask_s, 'subU1'],
                     df_sub.loc[mask_s, 'subU2'],
                     c=clr, s=10, alpha=0.75, rasterized=True,
                     linewidths=0, zorder=5)
    handles_sub.append(
        mpatches.Patch(color=clr, label=f'Sub-cluster {sid}  (n={n_s:,})')
    )

if noise_mask_s.any():
    handles_sub.append(
        mpatches.Patch(color=NOISE_COLOR,
                       label=f'No asignado  (n={noise_mask_s.sum():,})')
    )

ax_right.legend(handles=handles_sub, loc='lower right', fontsize=8,
                frameon=True, framealpha=0.9, edgecolor=ACCENT_COLOR,
                title='Sub-clusters', title_fontsize=8.5)

ax_right.set_xlabel('Sub-UMAP dim. 1', fontsize=9)
ax_right.set_ylabel('Sub-UMAP dim. 2', fontsize=9)
ax_right.set_title('b)  Después del subclustering\n'
                    f'Cluster 0 → {len(valid_sub_ids)} sub-tipos de sonido diferenciados',
                    fontsize=10, fontweight='bold', color=ACCENT_COLOR, pad=7)
ax_right.spines['top'].set_visible(False)
ax_right.spines['right'].set_visible(False)
ax_right.tick_params(labelsize=8)
for spine in ax_right.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color(ACCENT_COLOR)

# ── Título general ─────────────────────────────────────────────────────────────
fig.suptitle(
    'Subclustering jerárquico: refinamiento automático de clusters heterogéneos\n'
    'PA-17 (Pastizal) — Tapytá, Paraguay | Campaña Noviembre–Diciembre 2024',
    fontsize=11, y=0.97, color='#222222'
)

fig.text(0.5, 0.01,
         'El Cluster 0 del agrupamiento global agrupa segmentos acústicamente '
         'heterogéneos. El subclustering local los separa en 7 tipos diferenciados.',
         ha='center', fontsize=8, color='#666666', style='italic')

# ── Guardar ────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'FigC_Subclustering_AntesDespues.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✓ Figura guardada: {out_path}")
print("✓ Script C completado.")
