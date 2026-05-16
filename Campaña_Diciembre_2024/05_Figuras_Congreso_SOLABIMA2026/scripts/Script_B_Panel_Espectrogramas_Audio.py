# =============================================================================
# Script_B_Panel_Espectrogramas_Audio.py
# Campaña Nov-Dic 2024 — Paisajes Sonoros Tapytá
# Figura B: Panel 2×2 de espectrogramas — 4 arquetipos acústicos
# + exportación de clips de audio .wav para escucha
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_RESULTS = r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\Campaña_Diciembre_2024\02_Resultados_por_sitio'
SEAGATE_BASE = r'E:\Campaña diciembre 2024'
OUT_DIR = r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\Campaña_Diciembre_2024\05_Figuras_Congreso_SOLABIMA2026\FigB_Espectrogramas_Audio'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Definición de los 4 arquetipos acústicos ──────────────────────────────────
# Identificados visualmente inspeccionando clusters de PA-17Tapyta
# Clusters más periféricos (islas en UMAP) = eventos específicos
# Clusters más grandes y centrales = fondos continuos
ARCHETYPES = [
    {
        'site':      'PA-17Tapyta',
        'cluster':   2,
        'label':     'Arquetipo 1: Vocalización de anfibio',
        'sublabel':  '(llamados tónales repetitivos)',
        'color':     '#1565c0',   # azul
        'icon':      '🐸',
    },
    {
        'site':      'PA-17Tapyta',
        'cluster':   17,
        'label':     'Arquetipo 2: Coro de insectos nocturnos',
        'sublabel':  '(estridulación broadband continua)',
        'color':     '#558b2f',   # verde
        'icon':      '🦗',
    },
    {
        'site':      'PA-17Tapyta',
        'cluster':   11,
        'label':     'Arquetipo 3: Ruido abiótico',
        'sublabel':  '(lluvia — energía difusa sin estructura)',
        'color':     '#37474f',   # gris azulado
        'icon':      '🌧️',
    },
    {
        'site':      'PA-17Tapyta',
        'cluster':   0,
        'label':     'Arquetipo 4: Sonido mixto',
        'sublabel':  '(superposición de múltiples fuentes)',
        'color':     '#6a1b9a',   # violeta
        'icon':      '🔀',
    },
]

# ── Parámetros espectrales ─────────────────────────────────────────────────────
SR_TARGET   = 22050   # Hz de remuestreo
N_FFT       = 1024
HOP_LENGTH  = 256
FMAX        = 8000    # Hz máximo a mostrar
SEGMENT_SEC = 4.0     # segundos a mostrar

# ── Función: encontrar segmento representativo ─────────────────────────────────
def get_representative_segment(df_umap, cluster_id, n_candidates=5):
    """
    Devuelve el segmento más cercano al centroide del cluster.
    df_umap: DataFrame con DIM1, DIM2, cluster_hdbscan, archivo_origen, tiempo_inicio, tiempo_fin
    """
    sub = df_umap[df_umap['cluster_hdbscan'] == cluster_id].copy()
    if len(sub) == 0:
        return None
    centroid_x = sub['DIM1'].mean()
    centroid_y = sub['DIM2'].mean()
    sub['dist_centroide'] = np.sqrt((sub['DIM1'] - centroid_x)**2 +
                                     (sub['DIM2'] - centroid_y)**2)
    sub = sub.sort_values('dist_centroide')
    return sub.iloc[0]   # el más cercano al centroide = representativo

# ── Función: cargar audio desde Seagate ───────────────────────────────────────
def load_audio_segment(archivo_origen, t_start, t_end, seagate_base=SEAGATE_BASE):
    """
    Carga un segmento de audio desde la grabadora en el disco Seagate.
    archivo_origen: 'PA-17Tapyta/PA-17TAPYTA_20241201_194302.wav'
    """
    parts = archivo_origen.replace('\\', '/').split('/')
    site_folder = parts[0]
    wav_filename = parts[-1]
    wav_path = os.path.join(seagate_base, site_folder, 'Data', wav_filename)

    if not os.path.exists(wav_path):
        print(f"  ⚠ Archivo no encontrado: {wav_path}")
        return None, None

    try:
        y, sr = librosa.load(wav_path, sr=SR_TARGET,
                              offset=float(t_start), duration=SEGMENT_SEC,
                              mono=True)
        return y, sr
    except Exception as e:
        print(f"  ⚠ Error cargando {wav_path}: {e}")
        return None, None

# ── Función: generar espectrograma ─────────────────────────────────────────────
def compute_spectrogram(y, sr):
    """Devuelve espectrograma en dB, eje de frecuencias y tiempos."""
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]),
                                    sr=sr, hop_length=HOP_LENGTH)
    # Filtrar hasta FMAX
    freq_mask = freqs <= FMAX
    return S_db[freq_mask, :], freqs[freq_mask], times

# ── Cargar datos UMAP de PA-17 ────────────────────────────────────────────────
print("Cargando datos UMAP de PA-17Tapyta...")
site_dir = os.path.join(BASE_RESULTS, 'PA-17Tapyta')
umap_csv = [f for f in os.listdir(site_dir)
            if 'fase2_umap_hdbscan.csv' in f and 'metrics' not in f][0]
df_umap = pd.read_csv(os.path.join(site_dir, umap_csv))
print(f"  {len(df_umap):,} segmentos cargados")

# ── Recopilar segmentos representativos y audio ────────────────────────────────
print("\nObteniendo segmentos representativos y cargando audio...")
archetype_data = []

for arch in ARCHETYPES:
    print(f"\n  Arquetipo: {arch['label'][:40]}... (cluster {arch['cluster']})")
    rep = get_representative_segment(df_umap, arch['cluster'])
    if rep is None:
        print("    ⚠ Cluster no encontrado")
        archetype_data.append(None)
        continue

    y, sr = load_audio_segment(rep['archivo_origen'],
                                rep['tiempo_inicio'], rep['tiempo_fin'])
    if y is None:
        archetype_data.append(None)
        continue

    S_db, freqs, times = compute_spectrogram(y, sr)
    print(f"    ✓ Audio cargado ({len(y)/sr:.1f} s) | "
          f"archivo: {rep['archivo_origen'].split('/')[-1]}")

    # Guardar clip de audio
    audio_filename = f"audio_{arch['label'][:20].replace(' ','_').replace(':','')}.wav"
    audio_path = os.path.join(OUT_DIR, audio_filename)
    sf.write(audio_path, y, sr, subtype='PCM_16')
    print(f"    ✓ Audio guardado: {audio_filename}")

    archetype_data.append({
        'arch':    arch,
        'y':       y,
        'sr':      sr,
        'S_db':    S_db,
        'freqs':   freqs,
        'times':   times,
        'archivo': rep['archivo_origen'].split('/')[-1],
        'audio_path': audio_path,
    })

# ── Construir figura 2×2 ───────────────────────────────────────────────────────
print("\nGenerando figura panel espectrogramas...")

fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor('white')

gs_main = gridspec.GridSpec(3, 2, figure=fig,
                             height_ratios=[1, 1, 0.07],
                             hspace=0.45, wspace=0.28,
                             top=0.90, bottom=0.05,
                             left=0.07, right=0.96)

panel_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for idx, (row_i, col_i) in enumerate(panel_positions):
    ax = fig.add_subplot(gs_main[row_i, col_i])

    if idx >= len(archetype_data) or archetype_data[idx] is None:
        ax.set_visible(False)
        continue

    d  = archetype_data[idx]
    arch = d['arch']

    # ── Espectrograma ──────────────────────────────────────────────────────────
    img = ax.pcolormesh(d['times'], d['freqs'] / 1000, d['S_db'],
                         shading='auto', cmap='magma',
                         vmin=-60, vmax=0, rasterized=True)

    # ── Colorbar individual ────────────────────────────────────────────────────
    cbar = plt.colorbar(img, ax=ax, pad=0.01, shrink=0.85, aspect=20)
    cbar.set_label('dB (rel. max)', fontsize=7.5, labelpad=3)
    cbar.ax.tick_params(labelsize=7)

    # ── Etiquetas de ejes ──────────────────────────────────────────────────────
    ax.set_xlabel('Tiempo (s)', fontsize=9, labelpad=3)
    ax.set_ylabel('Frecuencia (kHz)', fontsize=9, labelpad=3)
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, FMAX / 1000)
    ax.set_xlim(0, SEGMENT_SEC)

    # Marcas de frecuencia en kHz
    ax.set_yticks([0, 1, 2, 4, 6, 8])
    ax.set_yticklabels(['0', '1', '2', '4', '6', '8'])

    # ── Borde de color del arquetipo ───────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
        spine.set_color(arch['color'])

    # ── Título del panel ───────────────────────────────────────────────────────
    ax.set_title(f"{arch['label']}\n{arch['sublabel']}",
                 fontsize=9.5, fontweight='bold', color=arch['color'],
                 pad=5)

    # ── Fuente del archivo ─────────────────────────────────────────────────────
    ax.text(0.99, 0.03, d['archivo'], transform=ax.transAxes,
            fontsize=6, ha='right', va='bottom', color='white',
            style='italic', alpha=0.75)

    # ── Letra del panel ────────────────────────────────────────────────────────
    panel_letters = ['a)', 'b)', 'c)', 'd)']
    ax.text(-0.06, 1.07, panel_letters[idx], transform=ax.transAxes,
            fontsize=12, fontweight='bold', color='#333333', va='top')

# ── Título general ─────────────────────────────────────────────────────────────
fig.suptitle(
    'Cuatro arquetipos acústicos identificados en el paisaje sonoro nocturno\n'
    'PA-17 (Pastizal) — Tapytá, Paraguay | Campaña Noviembre–Diciembre 2024',
    fontsize=12, y=0.96, color='#222222'
)

fig.text(0.5, 0.022,
         'Espectrogramas generados con STFT (n_fft=1024, hop=256) | '
         'Segmentos de 4 s correspondientes al centroide de cada cluster HDBSCAN',
         ha='center', fontsize=7.5, color='#666666', style='italic')

# ── Guardar ────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'FigB_Panel_Espectrogramas_4Arquetipos.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✓ Figura guardada: {out_path}")

# ── Resumen de audios generados ────────────────────────────────────────────────
print("\n── Archivos de audio generados ──────────────────")
for d in archetype_data:
    if d is not None:
        print(f"  {os.path.basename(d['audio_path'])}")

print("\n✓ Script B completado.")
