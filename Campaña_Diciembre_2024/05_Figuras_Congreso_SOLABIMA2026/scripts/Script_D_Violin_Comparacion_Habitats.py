# =============================================================================
# Script_D_Violin_Comparacion_Habitats.py
# Campaña Nov-Dic 2024 — Paisajes Sonoros Tapytá
# Figura D: Comparación de métricas acústicas por tipo de hábitat
# (violin plots + puntos individuales + anotaciones)
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
BASE_RESULTS = r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\Campaña_Diciembre_2024\02_Resultados_por_sitio'
OUT_DIR = r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\Campaña_Diciembre_2024\05_Figuras_Congreso_SOLABIMA2026\FigD_Violin_Habitats'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Mapa sitio → hábitat ───────────────────────────────────────────────────────
HABITAT_MAP = {
    'BO-31Tapyta': 'Bosque', 'BO-32Tapyta': 'Bosque', 'BO-33Tapyta': 'Bosque',
    'BO-34 Tapyta': 'Bosque', 'BO-35Tapyta': 'Bosque', 'BO-36Tapyta': 'Bosque',
    'BO-37Tapyta': 'Bosque', 'Bo-38Tapyta': 'Bosque', 'BO-39Tapyta': 'Bosque',
    'BO-40Tapyta': 'Bosque',
    'EU-16Tapyta': 'Eucaliptal', 'EU-17Tapyta': 'Eucaliptal', 'EU-18Tapyta': 'Eucaliptal',
    'EU-19Tapyta': 'Eucaliptal', 'EU-20Tapyta': 'Eucaliptal',
    'PA-16Tapyta': 'Pastizal', 'PA-17Tapyta': 'Pastizal', 'PA-18Tapyta': 'Pastizal',
    'PA-19Tapyta': 'Pastizal', 'PA-20Tapyta': 'Pastizal',
}

COLORS = {
    'Bosque':     '#2e7d32',   # verde bosque
    'Eucaliptal': '#6a1b9a',   # violeta
    'Pastizal':   '#e65100',   # naranja tierra
}

ORDER = ['Bosque', 'Eucaliptal', 'Pastizal']

# ── Cargar métricas de los 20 sitios ──────────────────────────────────────────
rows = []
for site_dir in os.listdir(BASE_RESULTS):
    fpath = os.path.join(BASE_RESULTS, site_dir)
    if not os.path.isdir(fpath):
        continue
    csvs = [f for f in os.listdir(fpath) if 'fase2_umap_hdbscan_metrics' in f and f.endswith('.csv')]
    if not csvs:
        continue
    df = pd.read_csv(os.path.join(fpath, csvs[0]))
    row = df.iloc[0].to_dict()
    # Normalizar nombre del sitio
    site_name = row['site']
    row['habitat'] = HABITAT_MAP.get(site_name, 'Desconocido')
    row['proporcion_ruido_pct'] = row['proporcion_ruido'] * 100
    rows.append(row)

df_all = pd.DataFrame(rows)
df_all = df_all[df_all['habitat'] != 'Desconocido']
print(f"Sitios cargados: {len(df_all)}")
print(df_all.groupby('habitat')[['n_clusters','silhouette','proporcion_ruido_pct']].describe())

# ── Función auxiliar: violin + puntos + media ─────────────────────────────────
def violin_panel(ax, data_by_hab, metric, ylabel, title, ylim=None, fmt='.1f',
                 yformat_fn=None):
    """Dibuja un violin plot con puntos jittered, media y desviación."""
    positions = range(len(ORDER))
    parts_list = []
    for i, hab in enumerate(ORDER):
        vals = data_by_hab[hab]
        if len(vals) < 2:
            continue
        vp = ax.violinplot([vals], positions=[i], widths=0.55,
                           showmeans=False, showmedians=False, showextrema=False)
        for pc in vp['bodies']:
            pc.set_facecolor(COLORS[hab])
            pc.set_alpha(0.35)
            pc.set_edgecolor(COLORS[hab])
            pc.set_linewidth(1.5)
        parts_list.append(vp)

        # Puntos individuales con jitter
        jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter([i + j for j in jitter], vals,
                   color=COLORS[hab], s=55, zorder=5, alpha=0.85,
                   edgecolors='white', linewidths=0.7)

        # Media ± SD
        m, s = np.mean(vals), np.std(vals)
        ax.plot([i - 0.18, i + 0.18], [m, m],
                color='black', linewidth=2.2, zorder=6)
        ax.errorbar(i, m, yerr=s, fmt='none', color='black',
                    capsize=5, capthick=1.5, elinewidth=1.5, zorder=6)

        # Anotación media
        label_y = m + s + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03 if ylim is None else ylim[1] * 0.97
        val_str = (f'{m:{fmt}}' if yformat_fn is None else yformat_fn(m))
        ax.text(i, m + s * 1.35 + 0.02, val_str,
                ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                color=COLORS[hab])

    ax.set_xticks(list(positions))
    ax.set_xticklabels(ORDER, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=7)
    if ylim:
        ax.set_ylim(ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.tick_params(axis='y', labelsize=10)

# ── Construir datos por hábitat ────────────────────────────────────────────────
data_nclusters   = {h: df_all[df_all['habitat'] == h]['n_clusters'].tolist()         for h in ORDER}
data_silhouette  = {h: df_all[df_all['habitat'] == h]['silhouette'].tolist()          for h in ORDER}
data_ruido       = {h: df_all[df_all['habitat'] == h]['proporcion_ruido_pct'].tolist() for h in ORDER}

# ── Figura ──────────────────────────────────────────────────────────────────────
np.random.seed(42)

fig = plt.figure(figsize=(16, 6.5))
fig.patch.set_facecolor('white')
gs = GridSpec(1, 3, figure=fig, wspace=0.38, left=0.07, right=0.97, top=0.80, bottom=0.18)

# Panel 1: Número de clusters
ax1 = fig.add_subplot(gs[0])
violin_panel(ax1, data_nclusters, 'n_clusters',
             'Número de clusters', 'a)  Diversidad acústica\n(N.° de clusters HDBSCAN)',
             ylim=(-2, 48), fmt='.0f')

# Panel 2: Silhouette score
ax2 = fig.add_subplot(gs[1])
violin_panel(ax2, data_silhouette, 'silhouette',
             'Silhouette score', 'b)  Calidad del agrupamiento\n(Silhouette score)',
             ylim=(-0.05, 0.72), fmt='.3f')
ax2.axhline(0.40, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
ax2.text(2.48, 0.41, 'umbral\naceptable (0.40)', fontsize=7.5, color='gray', ha='right')

# Panel 3: Proporción de ruido
ax3 = fig.add_subplot(gs[2])
violin_panel(ax3, data_ruido, 'proporcion_ruido_pct',
             'Proporción de ruido (%)', 'c)  Segmentos no asignados\n(ruido HDBSCAN, %)',
             ylim=(-1, 28), fmt='.1f')

# ── Leyenda compartida ─────────────────────────────────────────────────────────
handles = [mpatches.Patch(facecolor=COLORS[h], label=f'{h} (n={len(data_nclusters[h])})',
                           edgecolor=COLORS[h], alpha=0.75) for h in ORDER]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10.5,
           frameon=True, framealpha=0.9, edgecolor='#cccccc',
           bbox_to_anchor=(0.52, 0.04))

# ── Título general ─────────────────────────────────────────────────────────────
fig.suptitle('Comparación de métricas acústicas por tipo de hábitat\n'
             'Campaña Noviembre–Diciembre 2024 | Bosque Atlántico Interior, Tapytá',
             fontsize=12, y=0.97, fontstyle='italic', color='#333333')

# ── Guardar ────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'FigD_Violin_Comparacion_Habitats.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✓ Figura guardada: {out_path}")
