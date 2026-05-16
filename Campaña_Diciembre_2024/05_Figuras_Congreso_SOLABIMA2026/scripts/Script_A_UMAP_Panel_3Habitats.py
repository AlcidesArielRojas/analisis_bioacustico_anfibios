# =============================================================================
# Script_A_UMAP_Panel_3Habitats.py
# Campaña Nov-Dic 2024 — Paisajes Sonoros Tapytá
# Figura A: UMAP + HDBSCAN — Panel 3 hábitats (Bosque / Eucaliptal / Pastizal)
# Versión mejorada: leyenda compacta fuera del gráfico, colores diferenciados,
# anotaciones de métricas clave.
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgba
import warnings
warnings.filterwarnings('ignore')

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_RESULTS = r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\Campaña_Diciembre_2024\02_Resultados_por_sitio'
OUT_DIR = r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\Campaña_Diciembre_2024\05_Figuras_Congreso_SOLABIMA2026\FigA_UMAP_Panel'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Sitios representativos (uno por hábitat) ───────────────────────────────────
# BO-31: Bosque más complejo (32 clusters, Sil=0.394)
# EU-16: Eucaliptal más limpio (2 clusters, Sil=0.591)
# PA-17: Pastizal más rico (40 clusters, Sil=0.474)
SITES = [
    ('BO-31Tapyta',  'Bosque',     '#2e7d32', 'BO-31'),
    ('EU-16Tapyta',  'Eucaliptal', '#6a1b9a', 'EU-16'),
    ('PA-17Tapyta',  'Pastizal',   '#e65100', 'PA-17'),
]

# ── Paleta de colores para clusters (hasta 40 clusters distintos) ─────────────
def make_cluster_colors(n_clusters):
    """Genera n colores cualitativos bien diferenciados."""
    # Combinamos tab20 + tab20b + tab20c para cubrir hasta ~60 clusters
    cmap1 = plt.cm.get_cmap('tab20', 20)
    cmap2 = plt.cm.get_cmap('tab20b', 20)
    cmap3 = plt.cm.get_cmap('Set1', 9)
    colors = []
    for i in range(20): colors.append(cmap1(i))
    for i in range(20): colors.append(cmap2(i))
    for i in range(9):  colors.append(cmap3(i))
    return colors[:n_clusters]

NOISE_COLOR  = '#cccccc'   # gris claro para ruido (-1)
NOISE_ALPHA  = 0.25
VALID_ALPHA  = 0.55
PT_SIZE      = 1.0         # tamaño de los puntos UMAP
LEGEND_PT_SIZE = 6         # tamaño del marcador en la leyenda

# ── Función principal de un panel UMAP ────────────────────────────────────────
def plot_umap_panel(ax, df, site_label, habitat, accent_color, metrics):
    """
    Dibuja el scatter UMAP en el eje ax.
    df: DataFrame con columnas DIM1, DIM2, cluster_hdbscan
    metrics: dict con n_clusters, silhouette, proporcion_ruido
    """
    clusters = sorted([c for c in df['cluster_hdbscan'].unique() if c >= 0])
    n_cl = len(clusters)
    pal  = make_cluster_colors(n_cl)
    cl_to_color = {cl: pal[i] for i, cl in enumerate(clusters)}

    # Primero ruido (debajo)
    noise = df[df['cluster_hdbscan'] == -1]
    if len(noise) > 0:
        ax.scatter(noise['DIM1'], noise['DIM2'],
                   c=NOISE_COLOR, s=PT_SIZE, alpha=NOISE_ALPHA, rasterized=True,
                   linewidths=0, label='_nolegend_')

    # Luego clusters válidos (encima)
    for cl in clusters:
        sub = df[df['cluster_hdbscan'] == cl]
        ax.scatter(sub['DIM1'], sub['DIM2'],
                   c=[cl_to_color[cl]] * len(sub),
                   s=PT_SIZE, alpha=VALID_ALPHA, rasterized=True, linewidths=0)

    # Estilo del eje
    ax.set_xlabel('UMAP dim. 1', fontsize=9, labelpad=2)
    ax.set_ylabel('UMAP dim. 2', fontsize=9, labelpad=2)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Recuadro de hábitat (color del hábitat como borde)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color(accent_color)

    # Título con nombre del sitio y hábitat
    ax.set_title(f'{site_label} | {habitat}', fontsize=11, fontweight='bold',
                 color=accent_color, pad=6)

    # Cuadro de métricas clave en esquina inferior derecha
    stats_text = (f"Clusters: {metrics['n_clusters']}\n"
                  f"Silhouette: {metrics['silhouette']:.3f}\n"
                  f"Ruido: {metrics['proporcion_ruido']*100:.1f}%")
    ax.text(0.98, 0.03, stats_text, transform=ax.transAxes,
            fontsize=7.5, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=accent_color, alpha=0.85, linewidth=1.2))

    return clusters, cl_to_color

def build_legend(ax_leg, clusters, cl_to_color, accent_color):
    """Construye una leyenda compacta de clusters debajo del eje."""
    ax_leg.set_visible(False)
    n = len(clusters)
    ncols = min(8, n)  # máximo 8 columnas
    nrows = int(np.ceil(n / ncols))
    handles = []
    for cl in clusters:
        patch = mpatches.Patch(color=cl_to_color[cl],
                               label=f'C{cl}', linewidth=0)
        handles.append(patch)
    # Leyenda en el panel invisible
    leg = ax_leg.legend(handles=handles, loc='center', ncol=ncols,
                        fontsize=6.5, frameon=True,
                        framealpha=0.95, edgecolor=accent_color,
                        title=f'Clusters ({n} tipos de sonido)',
                        title_fontsize=7.5,
                        handlelength=1.2, handleheight=0.9,
                        columnspacing=0.6, handletextpad=0.4,
                        borderpad=0.5)
    ax_leg.set_visible(True)
    for spine in ax_leg.spines.values():
        spine.set_visible(False)
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])

# ── Cargar datos ──────────────────────────────────────────────────────────────
print("Cargando datos UMAP...")
site_data = {}
for site_id, habitat, color, label in SITES:
    site_dir = os.path.join(BASE_RESULTS, site_id)
    umap_csv = [f for f in os.listdir(site_dir)
                if 'fase2_umap_hdbscan.csv' in f and 'metrics' not in f]
    metrics_csv = [f for f in os.listdir(site_dir)
                   if 'fase2_umap_hdbscan_metrics.csv' in f]
    if not umap_csv or not metrics_csv:
        print(f"  AVISO: archivos faltantes para {site_id}")
        continue
    df_umap = pd.read_csv(os.path.join(site_dir, umap_csv[0]),
                          usecols=['DIM1', 'DIM2', 'cluster_hdbscan'])
    df_metrics = pd.read_csv(os.path.join(site_dir, metrics_csv[0]))
    metrics_dict = df_metrics.iloc[0].to_dict()
    site_data[site_id] = dict(df=df_umap, metrics=metrics_dict,
                               habitat=habitat, color=color, label=label)
    print(f"  ✓ {site_id}: {len(df_umap):,} puntos, "
          f"{int(metrics_dict['n_clusters'])} clusters")

# ── Construir figura combinada ─────────────────────────────────────────────────
print("\nGenerando figura combinada (3 hábitats)...")

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('white')

# Grilla: 2 filas × 3 columnas
# Fila 0 (alta) = UMAP scatter
# Fila 1 (baja) = Leyenda de clusters
gs = GridSpec(2, 3, figure=fig,
              height_ratios=[5.5, 1.0],
              hspace=0.20, wspace=0.32,
              top=0.90, bottom=0.06,
              left=0.05, right=0.98)

for col_idx, (site_id, habitat, color, label) in enumerate(SITES):
    if site_id not in site_data:
        continue
    d = site_data[site_id]
    ax_umap = fig.add_subplot(gs[0, col_idx])
    ax_leg  = fig.add_subplot(gs[1, col_idx])

    clusters, cl_to_color = plot_umap_panel(
        ax_umap, d['df'], label, habitat, color, d['metrics'])
    build_legend(ax_leg, clusters, cl_to_color, color)

# ── Título y subtítulo general ─────────────────────────────────────────────────
fig.suptitle(
    'Paisaje sonoro nocturno de tres tipos de hábitat — Tapytá, Paraguay\n'
    'Proyección UMAP + clustering HDBSCAN | Campaña Noviembre–Diciembre 2024',
    fontsize=12, y=0.97, color='#222222'
)

# Nota metodológica al pie
fig.text(0.5, 0.002,
         'Cada punto representa un segmento de 4 s de audio. '
         'Puntos grises = segmentos no asignados (ruido HDBSCAN).',
         ha='center', fontsize=8, color='#666666', style='italic')

# ── Guardar ────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'FigA_UMAP_Panel_3Habitats.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✓ Figura combinada guardada: {out_path}")

# ── También generar figuras individuales por sitio ────────────────────────────
print("\nGenerando figuras individuales...")
for site_id, habitat, color, label in SITES:
    if site_id not in site_data:
        continue
    d = site_data[site_id]
    n_cl = int(d['metrics']['n_clusters'])

    # Ajustar tamaño según número de clusters
    fig_h = 6.5 if n_cl <= 10 else 8.5
    fig_w = 8 if n_cl <= 10 else 10

    fig_i = plt.figure(figsize=(fig_w, fig_h))
    fig_i.patch.set_facecolor('white')

    gs_i = GridSpec(2, 1, figure=fig_i,
                    height_ratios=[5.5, 1.0 if n_cl <= 10 else 1.8],
                    hspace=0.22, top=0.88, bottom=0.06)
    ax_u = fig_i.add_subplot(gs_i[0])
    ax_l = fig_i.add_subplot(gs_i[1])

    clusters_i, cl_to_color_i = plot_umap_panel(
        ax_u, d['df'], label, habitat, color, d['metrics'])
    build_legend(ax_l, clusters_i, cl_to_color_i, color)

    fig_i.suptitle(
        f'{habitat} · {label} — UMAP + HDBSCAN\n'
        f'Campaña Nov–Dic 2024 | Tapytá, Paraguay',
        fontsize=11, y=0.96, color=color
    )
    out_i = os.path.join(OUT_DIR, f'FigA_{site_id}_UMAP_mejorado.png')
    fig_i.savefig(out_i, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {out_i}")

print("\n✓ Script A completado.")
