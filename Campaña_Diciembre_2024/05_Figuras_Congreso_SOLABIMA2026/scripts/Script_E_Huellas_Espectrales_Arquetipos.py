# =============================================================================
# Script_E_Huellas_Espectrales_Arquetipos.py
# Campaña Nov-Dic 2024 — Paisajes Sonoros Tapytá
# Figura E: Huellas espectrales medias de los 4 arquetipos acústicos
#
# Responde: "En qué me basé para identificar cada arquetipo?"
# Método: para cada cluster, carga N segmentos aleatorios, calcula
# el espectro de potencia medio (FFT) y lo grafica con banda de SD.
# Las diferencias entre perfiles son la evidencia objetiva.
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
BASE_RESULTS = r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\Campaña_Diciembre_2024\02_Resultados_por_sitio'
SEAGATE_BASE = r'E:\Campaña diciembre 2024'
OUT_DIR      = r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\Campaña_Diciembre_2024\05_Figuras_Congreso_SOLABIMA2026\FigE_Huellas_Espectrales'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Arquetipos a comparar ─────────────────────────────────────────────────────
ARCHETYPES = [
    {'cluster': 2,  'label': 'Arquetipo 1\nVocalizacion de anfibio',
     'color': '#1565c0', 'desc': 'Llamados tónicos repetitivos (1–3 kHz)'},
    {'cluster': 17, 'label': 'Arquetipo 2\nCoro de insectos nocturnos',
     'color': '#2e7d32', 'desc': 'Estridulacion broadband continua (2–8 kHz)'},
    {'cluster': 11, 'label': 'Arquetipo 3\nRuido abiotico (lluvia)',
     'color': '#37474f', 'desc': 'Energia plana sin estructura tonal'},
    {'cluster': 0,  'label': 'Arquetipo 4\nSonido mixto',
     'color': '#6a1b9a', 'desc': 'Superposicion de multiples fuentes'},
]

# ── Parámetros de audio ────────────────────────────────────────────────────────
SR_TARGET   = 22050
N_FFT       = 2048      # mayor resolucion espectral para el perfil medio
HOP_LENGTH  = 512
FMAX        = 8000
N_SAMPLES   = 25        # segmentos por arquetipo (balanceo computacion/precision)
SEGMENT_SEC = 4.0

np.random.seed(42)

# ── Cargar UMAP CSV de PA-17 ───────────────────────────────────────────────────
print("Cargando datos PA-17Tapyta...")
site_dir = os.path.join(BASE_RESULTS, 'PA-17Tapyta')
umap_csv = [f for f in os.listdir(site_dir)
            if 'fase2_umap_hdbscan.csv' in f and 'metrics' not in f][0]
df = pd.read_csv(os.path.join(site_dir, umap_csv))
print(f"  {len(df):,} segmentos totales")

# ── Función: cargar audio ──────────────────────────────────────────────────────
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

# ── Función: espectro de potencia medio ───────────────────────────────────────
def mean_power_spectrum(ys, sr=SR_TARGET, n_fft=N_FFT):
    """Devuelve frecuencias (Hz) y espectro medio (dB) sobre N segmentos."""
    spectra = []
    for y in ys:
        if y is None or len(y) < n_fft:
            continue
        D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=HOP_LENGTH))
        # Media temporal: perfil espectral de un segmento
        spec = D.mean(axis=1)
        spectra.append(spec)
    if not spectra:
        return None, None, None
    spectra = np.array(spectra)                        # (N, n_freqs)
    freqs   = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mean_db = librosa.amplitude_to_db(spectra.mean(axis=0), ref=np.max)
    sd_db   = spectra.std(axis=0)
    # Convertir SD a escala dB relativa (aproximacion lineal)
    mean_lin   = spectra.mean(axis=0)
    sd_db_true = 20 * np.log10(np.maximum(mean_lin + sd_db, 1e-10)) - \
                 20 * np.log10(np.maximum(mean_lin, 1e-10))
    return freqs, mean_db, sd_db_true

# ── Recopilar perfiles espectrales ────────────────────────────────────────────
print("\nCalculando huellas espectrales...")
arch_results = []

for arch in ARCHETYPES:
    cid = arch['cluster']
    sub = df[df['cluster_hdbscan'] == cid].copy()
    n_avail = len(sub)
    n_load  = min(N_SAMPLES, n_avail)
    sample  = sub.sample(n=n_load, random_state=42)
    print(f"\n  Cluster {cid} ({arch['label'][:30]}): {n_avail} segs, cargando {n_load}...")

    ys = []
    ok = 0
    for _, row in sample.iterrows():
        y = load_audio(row['archivo_origen'], row['tiempo_inicio'])
        ys.append(y)
        if y is not None:
            ok += 1
    print(f"    Audio cargado: {ok}/{n_load}")

    freqs, mean_db, sd_db = mean_power_spectrum(ys)
    arch_results.append({'arch': arch, 'freqs': freqs,
                         'mean_db': mean_db, 'sd_db': sd_db,
                         'n_ok': ok})

# ── Construir figura ───────────────────────────────────────────────────────────
print("\nGenerando figura de huellas espectrales...")

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('white')

# Layout: 2×2 arriba (perfiles individuales) + 1 fila abajo (comparacion)
gs = gridspec.GridSpec(3, 2, figure=fig,
                       height_ratios=[1, 1, 1.1],
                       hspace=0.55, wspace=0.35,
                       top=0.91, bottom=0.06, left=0.07, right=0.97)

panel_letters = ['a)', 'b)', 'c)', 'd)']

# Frecuencia maxima a graficar
freq_max = FMAX

for idx, res in enumerate(arch_results):
    row_i = idx // 2
    col_i = idx % 2
    ax = fig.add_subplot(gs[row_i, col_i])

    arch   = res['arch']
    freqs  = res['freqs']
    mdb    = res['mean_db']
    sdb    = res['sd_db']
    color  = arch['color']

    if freqs is None:
        ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                transform=ax.transAxes)
        continue

    # Filtrar hasta FMAX
    mask = freqs <= freq_max
    f_kHz = freqs[mask] / 1000
    m     = mdb[mask]
    s     = sdb[mask]

    # Banda de incertidumbre (SD)
    ax.fill_between(f_kHz, m - s, m + s,
                    alpha=0.18, color=color, label='±1 SD')

    # Perfil medio
    ax.plot(f_kHz, m, color=color, linewidth=2.0, label='Espectro medio')

    # Anotacion de pico dominante
    peak_idx  = np.argmax(m)
    peak_freq = f_kHz[peak_idx]
    peak_db   = m[peak_idx]
    ax.annotate(f'{peak_freq:.1f} kHz\n(pico)',
                xy=(peak_freq, peak_db),
                xytext=(peak_freq + 0.5, peak_db + 2),
                fontsize=7.5, color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                ha='left')

    ax.set_xlabel('Frecuencia (kHz)', fontsize=9, labelpad=3)
    ax.set_ylabel('Potencia espectral (dB)', fontsize=9, labelpad=3)
    ax.set_title(f'{arch["label"]}\n{arch["desc"]}',
                 fontsize=9.5, fontweight='bold', color=color, pad=5)
    ax.set_xlim(0, freq_max / 1000)
    ax.set_ylim(-50, 5)
    ax.grid(linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

    # Borde de color
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color(color)

    # Letra del panel
    ax.text(-0.12, 1.04, panel_letters[idx], transform=ax.transAxes,
            fontsize=11, fontweight='bold', color='#333333', va='top')

    # N de segmentos
    ax.text(0.98, 0.04, f'n = {res["n_ok"]} segmentos',
            transform=ax.transAxes, fontsize=7.5, ha='right',
            color='gray', style='italic')

# ── Panel inferior: superposición de los 4 perfiles (comparacion directa) ─────
ax_cmp = fig.add_subplot(gs[2, :])   # ocupa todo el ancho

for res in arch_results:
    if res['freqs'] is None:
        continue
    arch  = res['arch']
    mask  = res['freqs'] <= freq_max
    f_kHz = res['freqs'][mask] / 1000
    m     = res['mean_db'][mask]
    s     = res['sd_db'][mask]
    ax_cmp.fill_between(f_kHz, m - s, m + s,
                        alpha=0.12, color=arch['color'])
    ax_cmp.plot(f_kHz, m, color=arch['color'], linewidth=2.2,
                label=arch['label'].replace('\n', ' '))

ax_cmp.set_xlabel('Frecuencia (kHz)', fontsize=10, labelpad=4)
ax_cmp.set_ylabel('Potencia espectral media (dB)', fontsize=10, labelpad=4)
ax_cmp.set_title('e)  Comparacion directa: los 4 arquetipos superpuestos',
                 fontsize=11, fontweight='bold', color='#222222', pad=6)
ax_cmp.set_xlim(0, freq_max / 1000)
ax_cmp.set_ylim(-50, 5)
ax_cmp.grid(linestyle='--', alpha=0.35)
ax_cmp.spines['top'].set_visible(False)
ax_cmp.spines['right'].set_visible(False)
ax_cmp.tick_params(labelsize=9)
ax_cmp.legend(loc='lower right', fontsize=8.5, frameon=True,
              framealpha=0.9, ncol=2)

# Anotaciones de regiones clave
ax_cmp.axvspan(0.5, 2.5, alpha=0.07, color='#1565c0',
               label='Zona vocal anfibios')
ax_cmp.text(1.5, 3.5, 'Zona\nanfibios', fontsize=7.5,
            color='#1565c0', ha='center', alpha=0.8)
ax_cmp.axvspan(3.5, 8.0, alpha=0.07, color='#2e7d32')
ax_cmp.text(5.8, 3.5, 'Zona\ninsectos', fontsize=7.5,
            color='#2e7d32', ha='center', alpha=0.8)

# ── Titulo general ─────────────────────────────────────────────────────────────
fig.suptitle(
    'Huellas espectrales de los 4 arquetipos acusticos — PA-17 (Pastizal) | Tapyta\n'
    'Espectro de potencia medio (N=25 segmentos por arquetipo)',
    fontsize=11, y=0.98, color='#222222'
)

# ── Guardar ────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'FigE_Huellas_Espectrales_4Arquetipos.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nFigura guardada: {out_path}")
print("Script E completado.")
