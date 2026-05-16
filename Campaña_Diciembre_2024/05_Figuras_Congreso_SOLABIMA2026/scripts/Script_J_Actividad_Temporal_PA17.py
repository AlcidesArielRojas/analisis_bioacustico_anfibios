# =============================================================================
# Script_J_Actividad_Temporal_PA17.py
# Campaña Nov-Dic 2024 — Paisajes Sonoros Tapytá
#
# ¿QUÉ HACE ESTE SCRIPT?
# Extrae la hora exacta de cada segmento de audio (combinando la fecha/hora
# del nombre del archivo con el tiempo_inicio) y cuenta cuántos segmentos
# de cada categoría acústica ocurren en cada hora de la noche (18:00–06:00).
#
# CATEGORÍAS ACÚSTICAS (identificación manual confirmada):
#   Anfibio        → Clusters 1, 2, 3
#   Ave            → Cluster 39
#   Insecto        → Clusters 17, 30, 32, 34, 35, 36, 37
#   Lluvia/Tormenta→ Cluster 38
#   Mixto Vocal    → Cluster 0 (anfibio + insecto)
#
# SALIDAS:
#   FigJ_Actividad_Temporal_PA17.png  — 3 paneles:
#     A: Actividad de todas las categorías por hora (curvas suavizadas)
#     B: Los 3 clusters de anfibio por separado (¿mismos horarios = misma especie?)
#     C: Composición proporcional de la noche (gráfico de dona)
#
# CÓMO LEER LOS RESULTADOS:
#   - Eje X = hora de la noche (18:00 a 06:00; 00:00 = medianoche)
#   - Eje Y = número de segmentos de 4 s detectados en esa hora
#   - Un pico en la curva = hora de máxima actividad de ese tipo de sonido
#   - Si los 3 clusters anfibio tienen picos en horas diferentes → posibles
#     especies distintas con ritmos de canto distintos
# =============================================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_RESULTS = r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\Campaña_Diciembre_2024\02_Resultados_por_sitio'
OUT_DIR      = (r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local'
                r'\Campaña_Diciembre_2024\05_Figuras_Congreso_SOLABIMA2026'
                r'\FigJ_Actividad_Temporal')
os.makedirs(OUT_DIR, exist_ok=True)
SITE = 'PA-17Tapyta'

np.random.seed(42)

# ── Definición de categorías acústicas (identificación manual) ─────────────────
# Cada categoría agrupa los clusters con el mismo tipo de sonido.
# Los colores son los mismos que usamos en todas las figuras del proyecto.
CATEGORIES = {
    'Anfibio':         {'clusters': [1, 2, 3],             'color': '#1565c0'},
    'Ave':             {'clusters': [39],                   'color': '#e53935'},
    'Insecto':         {'clusters': [17, 30, 32, 34, 35, 36, 37], 'color': '#2e7d32'},
    'Lluvia/Tormenta': {'clusters': [38],                   'color': '#546e7a'},
    'Mixto Vocal':     {'clusters': [0],                    'color': '#6a1b9a'},
}

# Los 3 clusters de anfibio se analizan individualmente en el Panel B
FROG_CLUSTERS = {
    1: {'label': 'Cluster 1  (n=1,450)', 'color': '#64b5f6'},  # azul claro
    2: {'label': 'Cluster 2  (n=1,454)', 'color': '#1565c0'},  # azul medio
    3: {'label': 'Cluster 3  (n=5,809)', 'color': '#0d47a1'},  # azul oscuro
}

# ── Cargar datos de PA-17 ────────────────────────────────────────────────────
print(f"Cargando datos de {SITE}...")
site_dir = os.path.join(BASE_RESULTS, SITE)
umap_csv = [f for f in os.listdir(site_dir)
            if 'fase2_umap_hdbscan.csv' in f and 'metrics' not in f][0]
df = pd.read_csv(os.path.join(site_dir, umap_csv))
print(f"  {len(df):,} segmentos cargados")

# ── Extraer hora de grabación de cada segmento ────────────────────────────────
# El nombre del archivo tiene formato: PA-17TAPYTA_YYYYMMDD_HHMMSS.wav
# Ejemplo: PA-17TAPYTA_20241130_215802.wav → grabación inició a las 21:58:02
# Al sumarle tiempo_inicio (segundos), obtenemos la hora exacta del segmento.
print("Extrayendo timestamps de los archivos de audio...")

fname_series = df['archivo_origen'].str.replace('\\', '/', regex=False).str.split('/').str[-1]
pat = r'_(\d{8})_(\d{6})\.'
extracted = fname_series.str.extract(pat)
df['_rec_dt'] = pd.to_datetime(
    extracted[0] + '_' + extracted[1],
    format='%Y%m%d_%H%M%S',
    errors='coerce'
)
df['_seg_dt'] = df['_rec_dt'] + pd.to_timedelta(df['tiempo_inicio'].astype(float), unit='s')
df['hour']   = df['_seg_dt'].dt.hour
df['minute'] = df['_seg_dt'].dt.minute

# Mapa de hora a "hora nocturna": 18-23 quedan igual, 0-6 se desplazan a 24-30
# Esto permite un eje continuo de 18 (anochecer) a 30 (= 06:00 amanecer)
df['night_hour'] = df['hour'].apply(lambda h: h if h >= 18 else h + 24)

valid_ts = df['_seg_dt'].notna() & df['night_hour'].between(18, 30)
print(f"  Segmentos con hora nocturna válida: {valid_ts.sum():,} / {len(df):,}")

# ── Asignar categoría acústica a cada segmento ────────────────────────────────
cluster_to_cat = {}
for cat, info in CATEGORIES.items():
    for cid in info['clusters']:
        cluster_to_cat[cid] = cat

df['category'] = df['cluster_hdbscan'].map(cluster_to_cat)
df_known = df[valid_ts & df['category'].notna()].copy()

print(f"\nSegmentos por categoría:")
for cat, info in CATEGORIES.items():
    n = (df_known['category'] == cat).sum()
    pct = n / len(df_known) * 100 if len(df_known) > 0 else 0
    print(f"  {cat:20}: {n:>5,}  ({pct:.1f}%)")

# ── Contar segmentos por hora y categoría ─────────────────────────────────────
HOURS = np.arange(18, 31)   # 18, 19, ..., 30 (30 = 06:00 del día siguiente)

def count_by_hour(df_sub):
    """Cuenta cuántos segmentos hay en cada hora nocturna."""
    return np.array([len(df_sub[df_sub['night_hour'] == h]) for h in HOURS], dtype=float)

counts_cat   = {cat: count_by_hour(df_known[df_known['category'] == cat])
                for cat in CATEGORIES}
counts_frog  = {cid: count_by_hour(df_known[df_known['cluster_hdbscan'] == cid])
                for cid in FROG_CLUSTERS}

# ── Función de suavizado (media móvil + interpolación) ───────────────────────
# Con solo 13 puntos horarios, suavizamos con ventana de 2h y luego
# interpolamos a una curva continua para mejorar la apariencia visual.
def smooth_and_interpolate(arr, window=3, n_points=300):
    s = pd.Series(arr, dtype=float)
    smoothed = s.rolling(window=window, center=True, min_periods=1).mean().values
    x_fine = np.linspace(HOURS[0], HOURS[-1], n_points)
    y_fine = np.interp(x_fine, HOURS, smoothed)
    y_fine = np.maximum(y_fine, 0)   # no negativos
    return x_fine, y_fine

# ── Etiquetas del eje X ───────────────────────────────────────────────────────
# Convierte el "número nocturno" de vuelta a etiqueta horaria legible
def hour_label(h):
    real = h if h < 24 else h - 24
    return f"{real:02d}:00"

TICK_HOURS  = list(range(18, 31, 2))   # cada 2 horas en el eje
TICK_LABELS = [hour_label(h) for h in TICK_HOURS]
TICK_LABELS[3] = "00:00\n(medianoche)"   # el tercer tick = hora 24

# ── Fases de la noche (para sombreado de fondo) ───────────────────────────────
PHASES = [
    (18, 20, '#ff9800', 'Anochecer\n18–20h'),
    (20, 24, '#3f51b5', 'Noche\n20–00h'),
    (24, 28, '#1a237e', 'Madrugada\n00–04h'),
    (28, 30, '#e64a19', 'Alba\n04–06h'),
]

def draw_phase_background(ax, ytext=0.96):
    """Pinta las franjas de fases de la noche y sus etiquetas."""
    for start, end, color, label in PHASES:
        ax.axvspan(start, end, alpha=0.07, color=color, zorder=0)
        ax.text((start + end) / 2, ytext, label,
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=8,
                color=color, fontweight='bold', alpha=0.9)
    ax.axvline(24, color='#555555', ls='--', lw=1.4, alpha=0.4, zorder=2)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerando figura de actividad temporal...")

fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(2, 3, figure=fig,
                       height_ratios=[1.3, 1],
                       hspace=0.52, wspace=0.32,
                       top=0.91, bottom=0.08,
                       left=0.07, right=0.97)

ax_main  = fig.add_subplot(gs[0, :])       # Panel A: todas las categorías
ax_frogs = fig.add_subplot(gs[1, 0:2])    # Panel B: anfibios en detalle
ax_donut = fig.add_subplot(gs[1, 2])      # Panel C: composición total (dona)

# ── Panel A: actividad de todas las categorías ────────────────────────────────
draw_phase_background(ax_main, ytext=0.96)

peak_info = {}
for cat, info in CATEGORIES.items():
    xf, yf = smooth_and_interpolate(counts_cat[cat])
    color   = info['color']
    ax_main.fill_between(xf, 0, yf, alpha=0.13, color=color, zorder=3)
    ax_main.plot(xf, yf, color=color, lw=2.4, label=cat, zorder=4)

    # Anotar la hora pico de cada categoría
    peak_idx_f = np.argmax(yf)
    peak_h_real = xf[peak_idx_f]
    peak_h_label = hour_label(int(round(peak_h_real)))
    ax_main.annotate(
        peak_h_label,
        xy=(xf[peak_idx_f], yf[peak_idx_f]),
        xytext=(0, 9), textcoords='offset points',
        fontsize=7.5, ha='center', color=color, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                  alpha=0.8, edgecolor=color, linewidth=0.8)
    )
    peak_info[cat] = (peak_h_label, int(counts_cat[cat].max()))

ax_main.set_xlim(18, 30)
ax_main.set_ylim(0)
ax_main.set_xticks(TICK_HOURS)
ax_main.set_xticklabels(TICK_LABELS, fontsize=9)
ax_main.set_ylabel('N.° de segmentos de 4 s por hora\n(curva suavizada, ventana 3h)', fontsize=10, labelpad=5)
ax_main.set_title(
    'a)  Actividad acústica nocturna por categoría — PA-17 (Pastizal) | Tapytá',
    fontsize=11, fontweight='bold', loc='left', pad=8)
ax_main.spines['top'].set_visible(False)
ax_main.spines['right'].set_visible(False)
ax_main.grid(axis='y', ls='--', alpha=0.3, zorder=1)
ax_main.tick_params(labelsize=9)
ax_main.legend(loc='upper right', fontsize=9.5, frameon=True,
               framealpha=0.93, edgecolor='#cccccc')

# ── Panel B: los 3 clusters de anfibio por separado ──────────────────────────
# Se normalizan a su propio máximo para comparar la FORMA temporal,
# independientemente de cuántos segmentos tenga cada cluster.
draw_phase_background(ax_frogs, ytext=0.97)

for cid, finfo in FROG_CLUSTERS.items():
    arr  = counts_frog[cid]
    xf, yf = smooth_and_interpolate(arr)
    # Normalizar 0–100%
    ymax = yf.max()
    yn   = (yf / ymax * 100) if ymax > 0 else yf
    color = finfo['color']
    ax_frogs.fill_between(xf, 0, yn, alpha=0.15, color=color, zorder=3)
    ax_frogs.plot(xf, yn, color=color, lw=2.4, label=finfo['label'], zorder=4)

    # Marcar la hora pico
    peak_x = xf[np.argmax(yn)]
    ax_frogs.axvline(peak_x, color=color, ls=':', lw=1.4, alpha=0.6, zorder=2)
    ax_frogs.text(peak_x, 103, hour_label(int(round(peak_x))),
                  ha='center', va='bottom', fontsize=8, color=color,
                  fontweight='bold')

ax_frogs.set_xlim(18, 30)
ax_frogs.set_ylim(0, 115)
ax_frogs.set_xticks(TICK_HOURS)
ax_frogs.set_xticklabels(TICK_LABELS, fontsize=8.5)
ax_frogs.set_ylabel('Actividad relativa (% de pico propio)', fontsize=10, labelpad=5)
ax_frogs.set_title(
    'b)  Comparación temporal de los 3 clusters de anfibio\n'
    '(curvas normalizadas al pico propio — para comparar CUÁNDO, no cuánto)',
    fontsize=10, fontweight='bold', loc='left', pad=7)
ax_frogs.spines['top'].set_visible(False)
ax_frogs.spines['right'].set_visible(False)
ax_frogs.grid(axis='y', ls='--', alpha=0.3)
ax_frogs.tick_params(labelsize=8.5)
ax_frogs.legend(loc='upper right', fontsize=9, frameon=True,
                framealpha=0.93, edgecolor='#cccccc',
                title='Cluster (normalizado a pico=100%)', title_fontsize=8.5)

# Nota explicativa
ax_frogs.text(0.01, 0.04,
    'Si los picos están a horas distintas → posibles especies diferentes.\n'
    'Si los picos coinciden → misma especie o respuesta al mismo estímulo.',
    transform=ax_frogs.transAxes, fontsize=7.5, color='#555555',
    style='italic', va='bottom')

# ── Panel C: gráfico de dona — composición total de la noche ─────────────────
# Muestra qué porcentaje del total de segmentos categorizados
# pertenece a cada tipo de sonido (visión global de la noche completa).
totals  = {cat: int(counts_cat[cat].sum()) for cat in CATEGORIES}
total_n = sum(totals.values())
sizes   = [totals[cat] for cat in CATEGORIES]
colors  = [CATEGORIES[cat]['color'] for cat in CATEGORIES]
labels  = [f"{cat}\n{totals[cat]:,}\n({totals[cat]/total_n*100:.1f}%)"
           for cat in CATEGORIES]

wedges, texts = ax_donut.pie(
    sizes, labels=None, colors=colors,
    startangle=90, counterclock=False,
    wedgeprops=dict(width=0.52, edgecolor='white', linewidth=1.5)
)

# Leyenda lateral
legend_labels = [f"{cat}  ({totals[cat]:,}  –  {totals[cat]/total_n*100:.1f}%)"
                 for cat in CATEGORIES]
ax_donut.legend(
    wedges, legend_labels,
    loc='lower center', bbox_to_anchor=(0.5, -0.28),
    fontsize=8, frameon=True, framealpha=0.92,
    edgecolor='#cccccc', ncol=1, handlelength=1.0
)

# Texto central de la dona
ax_donut.text(0, 0, f'{total_n:,}\nsegmentos\ncategorizados',
              ha='center', va='center', fontsize=8.5,
              fontweight='bold', color='#333333')

ax_donut.set_title(
    'c)  Composición total\nde la noche completa',
    fontsize=10, fontweight='bold', pad=10)

# ── Título general ─────────────────────────────────────────────────────────────
fig.suptitle(
    'Patrones de actividad acústica nocturna — PA-17 (Pastizal, Tapytá) | Campaña Nov–Dic 2024\n'
    'Categorización basada en identificación manual + clustering HDBSCAN sin supervisión',
    fontsize=12, y=0.98, fontweight='bold', color='#222222'
)

# ── Guardar ───────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'FigJ_Actividad_Temporal_PA17.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n  ✓ Figura guardada: {out_path}")

# ── Resumen estadístico ────────────────────────────────────────────────────────
print("\n" + "="*55)
print("RESUMEN — Patrones de actividad por categoría")
print("="*55)
for cat, info in CATEGORIES.items():
    arr  = counts_cat[cat]
    peak_h = HOURS[np.argmax(arr)]
    total  = int(arr.sum())
    pct    = total / total_n * 100 if total_n > 0 else 0
    print(f"  {cat:20}: total={total:>5,} segs ({pct:4.1f}%) | hora pico = {hour_label(peak_h)}")

print("\nComparación de clusters de anfibio:")
for cid, finfo in FROG_CLUSTERS.items():
    arr   = counts_frog[cid]
    peak_h = HOURS[np.argmax(arr)]
    total  = int(arr.sum())
    print(f"  {finfo['label']:30}: total={total:>5,} | hora pico = {hour_label(peak_h)}")
print("="*55)
print("\nScript J completado.")
