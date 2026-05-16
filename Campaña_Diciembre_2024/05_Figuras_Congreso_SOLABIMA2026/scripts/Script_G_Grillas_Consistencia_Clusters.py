# =============================================================================
# Script_G_Grillas_Consistencia_Clusters.py
# Campaña Nov-Dic 2024 — Paisajes Sonoros Tapytá
# Figuras G1/G2/G3: Consistencia de clusters acústicos por tipo de segmento
#
# Se generan 3 figuras separadas, cada una con 8 clusters (2×4 grid):
#   G1 — CENTROIDES:  el segmento más cercano al centroide de cada cluster
#   G2 — ALEATORIOS:  un segmento al azar (para verificar consistencia interna)
#   G3 — EXTREMOS:    el segmento más lejano del centroide (periferia del cluster)
#
# Ventaja: permite comparar directamente todos los clusters para un mismo
# tipo de segmento, en lugar de ver todo mezclado en un solo gráfico grande.
#
# Criterio de selección de clusters "importantes":
#   PA-17 (Pastizal) : clusters 2, 17, 11, 0 — los 4 arquetipos acústicos
#   EU-16 (Eucaliptal): clusters 0 y 1 — solo hay 2, muy contrastantes
#   BO-31 (Bosque)   : clusters 1 (el más grande) y 2 (el más compacto)
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
BASE_RESULTS = r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\Campaña_Diciembre_2024\02_Resultados_por_sitio'
SEAGATE_BASE = r'E:\Campaña diciembre 2024'
OUT_DIR      = (r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local'
                r'\Campaña_Diciembre_2024\05_Figuras_Congreso_SOLABIMA2026\FigG_Consistencia_Clusters')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Clusters seleccionados ────────────────────────────────────────────────────
# Cada entrada: (sitio, cluster_id, etiqueta corta, etiqueta larga, color borde, icono habitat)
SELECTED = [
    ('PA-17Tapyta', 2,
     'Vocaliz. anfibio\nPA-17 · Cl. 2',
     'PA-17 · Cl. 2\nVocalización de anfibio\n(llamados tónales 1–2 kHz)',
     '#1565c0', 'Pastizal'),
    ('PA-17Tapyta', 17,
     'Coro de insectos\nPA-17 · Cl. 17',
     'PA-17 · Cl. 17\nCoro de insectos nocturnos\n(estridulación broadband)',
     '#2e7d32', 'Pastizal'),
    ('PA-17Tapyta', 11,
     'Ruido abiótico\nPA-17 · Cl. 11',
     'PA-17 · Cl. 11\nRuido abiótico (lluvia)\n(energía difusa plana)',
     '#37474f', 'Pastizal'),
    ('PA-17Tapyta', 0,
     'Sonido mixto\nPA-17 · Cl. 0',
     'PA-17 · Cl. 0\nSonido mixto\n(candidato a subclustering)',
     '#6a1b9a', 'Pastizal'),
    ('EU-16Tapyta', 0,
     'Eucaliptal A\nEU-16 · Cl. 0',
     'EU-16 · Cl. 0\nEucaliptal — tipo A\n(n=4 582, Sil alta)',
     '#7b1fa2', 'Eucaliptal'),
    ('EU-16Tapyta', 1,
     'Eucaliptal B\nEU-16 · Cl. 1',
     'EU-16 · Cl. 1\nEucaliptal — tipo B\n(n=95 418, cluster masivo)',
     '#9c27b0', 'Eucaliptal'),
    ('BO-31Tapyta', 1,
     'Bosque principal\nBO-31 · Cl. 1',
     'BO-31 · Cl. 1\nBosque — sonido dominante\n(n=3 851)',
     '#2e7d32', 'Bosque'),
    ('BO-31Tapyta', 2,
     'Bosque compacto\nBO-31 · Cl. 2',
     'BO-31 · Cl. 2\nBosque — cluster compacto\n(n=811, muy homogéneo)',
     '#43a047', 'Bosque'),
]

# Colores de hábitat para el fondo del título de cada panel
HABITAT_BG = {
    'Pastizal':   '#fff3e0',   # naranja muy claro
    'Eucaliptal': '#f3e5f5',   # violeta muy claro
    'Bosque':     '#e8f5e9',   # verde muy claro
}
HABITAT_COLOR = {
    'Pastizal':   '#e65100',
    'Eucaliptal': '#6a1b9a',
    'Bosque':     '#2e7d32',
}

# ── Parámetros de audio ────────────────────────────────────────────────────────
SR_TARGET   = 22050
N_FFT       = 1024
HOP_LENGTH  = 256
FMAX        = 8000
SEGMENT_SEC = 4.0

np.random.seed(42)

# ── Cargar CSVs (con caché) ───────────────────────────────────────────────────
_csv_cache = {}

def get_site_df(site_id):
    if site_id not in _csv_cache:
        site_dir = os.path.join(BASE_RESULTS, site_id)
        fname    = [f for f in os.listdir(site_dir)
                    if 'fase2_umap_hdbscan.csv' in f and 'metrics' not in f][0]
        _csv_cache[site_id] = pd.read_csv(os.path.join(site_dir, fname))
        print(f"  CSV cargado: {site_id} ({len(_csv_cache[site_id]):,} filas)")
    return _csv_cache[site_id]

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
    mask = (freqs > 0) & (freqs <= FMAX)
    return S_db[mask, :], freqs[mask], times

def select_three(df_cl):
    """Devuelve (row_centroide, row_aleatorio, row_extremo)."""
    df = df_cl.copy()
    cx = df['DIM1'].mean()
    cy = df['DIM2'].mean()
    df['dist'] = np.sqrt((df['DIM1'] - cx)**2 + (df['DIM2'] - cy)**2)
    row_c = df.nsmallest(1, 'dist').iloc[0]
    row_e = df.nlargest(1, 'dist').iloc[0]
    pool  = df[(df.index != row_c.name) & (df.index != row_e.name)]
    row_a = pool.sample(1).iloc[0] if len(pool) > 0 else row_c
    return row_c, row_a, row_e

# ── Recopilar todos los espectrogramas ────────────────────────────────────────
print("Cargando datos y calculando espectrogramas...")

all_data = []   # lista de dicts, uno por cluster

for (site, cid, lbl_short, lbl_long, color, habitat) in SELECTED:
    print(f"\n→ {lbl_short.replace(chr(10),' | ')}")
    df_site = get_site_df(site)
    df_cl   = df_site[df_site['cluster_hdbscan'] == cid]
    n_seg   = len(df_cl)
    print(f"  n = {n_seg:,} segmentos")

    row_c, row_a, row_e = select_three(df_cl)

    specs = {}
    for tipo, row in [('centroide', row_c), ('aleatorio', row_a), ('extremo', row_e)]:
        y = load_audio(row['archivo_origen'], row['tiempo_inicio'])
        if y is not None and len(y) > N_FFT:
            specs[tipo] = compute_spec(y)
        else:
            specs[tipo] = None
            print(f"  ⚠ No se pudo cargar: {tipo}")

    all_data.append({
        'site':      site,
        'cluster':   cid,
        'lbl_short': lbl_short,
        'lbl_long':  lbl_long,
        'color':     color,
        'habitat':   habitat,
        'n_seg':     n_seg,
        'specs':     specs,
    })

# ── Función para construir y guardar una figura por tipo ──────────────────────
def build_figure(tipo, tipo_titulo, tipo_subtitulo, tipo_color_borde, filename):
    """
    tipo           : 'centroide' | 'aleatorio' | 'extremo'
    tipo_titulo    : texto principal del suptitle
    tipo_subtitulo : descripción breve para la nota al pie
    tipo_color_borde: color del marco de cada panel para ese tipo
    filename       : nombre de archivo de salida (sin ruta)
    """
    n_clusters = len(all_data)   # 8

    # Figura vertical: un cluster por fila, columna única
    fig = plt.figure(figsize=(11, n_clusters * 2.6))
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(n_clusters, 1, figure=fig,
                           hspace=0.55,
                           top=0.96, bottom=0.04,
                           left=0.09, right=0.97)

    panel_letters = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)']

    for idx, cd in enumerate(all_data):
        ax    = fig.add_subplot(gs[idx, 0])
        spec  = cd['specs'].get(tipo)
        color = cd['color']
        hab   = cd['habitat']

        if spec is not None:
            S_db, freqs, times = spec
            ax.pcolormesh(times, freqs, S_db,
                          shading='auto', cmap='magma',
                          vmin=-60, vmax=0, rasterized=True)
            ax.set_xlim(0, SEGMENT_SEC)
        else:
            ax.set_facecolor('#f5f5f5')
            ax.text(0.5, 0.5, 'No disponible',
                    ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='gray')

        # ── Eje Y logarítmico en Hz (como el espectrograma de referencia) ──────
        ax.set_yscale('log')
        ax.set_ylim(30, FMAX)
        ax.tick_params(labelsize=7.5)
        ax.set_yticks([100, 500, 1000, 2000, 4000, 8000])
        ax.set_yticklabels(['100', '500', '1k', '2k', '4k', '8k'])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylabel('Hz', fontsize=8.5, labelpad=3)

        if idx == n_clusters - 1:
            ax.set_xlabel('Tiempo (s)', fontsize=9, labelpad=3)
        else:
            ax.set_xticklabels([])

        # ── Borde fino del panel (color del cluster) ──────────────────────────
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
            spine.set_color(color)

        # ── Título limpio: letra + etiqueta + n ───────────────────────────────
        lbl   = cd['lbl_short'].replace('\n', ' · ')
        n_str = f'n = {cd["n_seg"]:,}'
        ax.set_title(
            f'{panel_letters[idx]}  {lbl}  ·  {n_str}',
            fontsize=9.5, fontweight='bold', color=color,
            pad=4, loc='left')

    # ── Leyenda de hábitats ────────────────────────────────────────────────────
    handles_hab = [
        mpatches.Patch(facecolor=HABITAT_COLOR[h], label=h,
                       edgecolor=HABITAT_COLOR[h], alpha=0.7)
        for h in ['Pastizal', 'Eucaliptal', 'Bosque']
    ]
    fig.legend(handles=handles_hab, loc='lower center', ncol=3,
               fontsize=10, frameon=True, framealpha=0.92,
               edgecolor='#cccccc', bbox_to_anchor=(0.5, 0.005),
               title='Tipo de hábitat', title_fontsize=9)

    # ── Título general ────────────────────────────────────────────────────────
    fig.suptitle(
        f'{tipo_titulo}\n'
        f'PA-17 (Pastizal) · EU-16 (Eucaliptal) · BO-31 (Bosque) | Tapytá, Paraguay',
        fontsize=12, y=0.99, color='#222222', fontweight='bold'
    )

    # ── Nota al pie ───────────────────────────────────────────────────────────
    fig.text(0.5, 0.018,
             tipo_subtitulo,
             ha='center', fontsize=8.5, color='#555555', style='italic')

    # ── Guardar ───────────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Guardada: {out_path}")


# ── Generar las 3 figuras ─────────────────────────────────────────────────────
print("\n--- Generando FigG1: CENTROIDES ---")
build_figure(
    tipo='centroide',
    tipo_titulo='G1 — Centroides: el segmento más representativo de cada cluster',
    tipo_subtitulo=(
        'Cada panel muestra el segmento cuyo vector MFCC es más cercano al centroide del cluster en el espacio UMAP 3D. '
        'Son los "ejemplares tipo" del cluster.'
    ),
    tipo_color_borde='solid',
    filename='FigG1_Centroides_8clusters.png'
)

print("\n--- Generando FigG2: ALEATORIOS ---")
build_figure(
    tipo='aleatorio',
    tipo_titulo='G2 — Aleatorios: muestra interna para verificar consistencia',
    tipo_subtitulo=(
        'Cada panel muestra un segmento elegido al azar dentro del cluster. '
        'Si se parece al centroide (FigG1), el cluster es internamente consistente.'
    ),
    tipo_color_borde='dashed',
    filename='FigG2_Aleatorios_8clusters.png'
)

print("\n--- Generando FigG3: EXTREMOS ---")
build_figure(
    tipo='extremo',
    tipo_titulo='G3 — Extremos: el segmento más lejano del centro de cada cluster',
    tipo_subtitulo=(
        'Cada panel muestra el segmento en la periferia del cluster (máxima distancia al centroide en UMAP). '
        'Cuanto más se parezca al centroide (FigG1), más homogéneo es el cluster.'
    ),
    tipo_color_borde='dotted',
    filename='FigG3_Extremos_8clusters.png'
)

print("\nScript G completado. 3 figuras generadas:")
for f in ['FigG1_Centroides_8clusters.png',
          'FigG2_Aleatorios_8clusters.png',
          'FigG3_Extremos_8clusters.png']:
    print(f"  {os.path.join(OUT_DIR, f)}")
