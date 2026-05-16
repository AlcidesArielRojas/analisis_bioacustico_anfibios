# =============================================================================
# Script_K_Comparacion_Espectral_Anfibios_PA17.py
# Campaña Nov-Dic 2024 — Paisajes Sonoros Tapytá
#
# Objetivo: Comparar las huellas espectrales de los tres clusters de
#   vocalización de anfibio identificados en PA-17:
#     Cluster 1  (n ≈ 1,450)
#     Cluster 2  (n ≈ 1,454)
#     Cluster 3  (n ≈ 5,809)
#
# Metodología:
#   Para cada cluster se seleccionan hasta N_SEG segmentos al azar.
#   De cada segmento se carga el audio (4 s, 22050 Hz) y se calcula
#   el espectro de potencia media (PSD) usando Welch / STFT-mean.
#   Se promedian todos los espectros del cluster → media + banda ±1 SD.
#
# Salida:
#   FigK_Comparacion_Espectral_Anfibios_PA17.png
#   Layout 2 × 2:
#     [A] Cluster 1 — espectro medio ± SD
#     [B] Cluster 2 — espectro medio ± SD
#     [C] Cluster 3 — espectro medio ± SD
#     [D] Overlay comparativo — 3 espectros medios superpuestos
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import librosa
import warnings
warnings.filterwarnings('ignore')

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_RESULTS = r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\Campaña_Diciembre_2024\02_Resultados_por_sitio'
SEAGATE_BASE = r'E:\Campaña diciembre 2024'
OUT_DIR      = (r'C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local'
                r'\Campaña_Diciembre_2024\05_Figuras_Congreso_SOLABIMA2026'
                r'\FigK_Comparacion_Espectral_Anfibios')
os.makedirs(OUT_DIR, exist_ok=True)

SITE        = 'PA-17Tapyta'
SR_TARGET   = 22050
SEGMENT_SEC = 4.0
N_FFT       = 2048          # ventana STFT → resolución 10.77 Hz
HOP_LENGTH  = 512
FMIN        = 100           # Hz — corte inferior del gráfico
FMAX        = 8000          # Hz — corte superior
N_SEG       = 250           # segmentos por cluster (máximo)
np.random.seed(42)

# Frog clusters confirmed by manual listening
FROG_CLUSTERS = {
    1: {'label': 'Cluster 1',  'color': '#64b5f6', 'linestyle': '--'},
    2: {'label': 'Cluster 2',  'color': '#1565c0', 'linestyle': '-'},
    3: {'label': 'Cluster 3',  'color': '#0d47a1', 'linestyle': ':'},
}

# ── Cargar CSV ────────────────────────────────────────────────────────────────
print(f"Cargando datos de {SITE}...")
site_dir = os.path.join(BASE_RESULTS, SITE)
umap_csv = [f for f in os.listdir(site_dir)
            if 'fase2_umap_hdbscan.csv' in f and 'metrics' not in f][0]
df = pd.read_csv(os.path.join(site_dir, umap_csv))
print(f"  {len(df):,} segmentos totales")

# ── Función: cargar audio de un segmento ─────────────────────────────────────
def load_audio(archivo_origen, t_start):
    parts    = str(archivo_origen).replace('\\', '/').split('/')
    wav_path = os.path.join(SEAGATE_BASE, parts[0], 'Data', parts[-1])
    if not os.path.exists(wav_path):
        return None
    try:
        y, _ = librosa.load(wav_path, sr=SR_TARGET,
                            offset=float(t_start),
                            duration=SEGMENT_SEC, mono=True)
        return y
    except Exception:
        return None

# ── Función: espectro de potencia medio de un clip ───────────────────────────
def power_spectrum(y, sr=SR_TARGET, n_fft=N_FFT, hop=HOP_LENGTH):
    """
    Devuelve (freqs [Hz], psd_db [dB]) para el clip y.
    Se calcula como la media temporal del espectrograma STFT (en dB).
    """
    S  = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, window='hann'))
    S2 = S ** 2
    # Media temporal → espectro de potencia
    psd = S2.mean(axis=1)
    # Convertir a dB (ref = valor máximo para normalizar entre clips)
    psd_db = 10 * np.log10(psd + 1e-12)
    freqs  = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return freqs, psd_db

# ── Función: normalizar espectro a máx = 0 dB ────────────────────────────────
def normalize_psd(psd_db, freqs, fmin=FMIN, fmax=FMAX):
    mask = (freqs >= fmin) & (freqs <= fmax)
    ref  = psd_db[mask].max()
    return psd_db - ref

# ── Calcular espectros por cluster ────────────────────────────────────────────
print("\nCalculando espectros de potencia por cluster...")
cluster_spectra = {}  # cid → {'freqs', 'mean', 'std', 'n'}

for cid, info in FROG_CLUSTERS.items():
    sub = df[df['cluster_hdbscan'] == cid].copy()
    n_total = len(sub)
    # Seleccionar muestra aleatoria
    n_sample = min(N_SEG, n_total)
    sub_samp = sub.sample(n=n_sample, random_state=42)

    print(f"  Cluster {cid}: {n_total:,} segs totales → procesando {n_sample}")

    all_psd = []
    n_ok = 0
    for _, row in sub_samp.iterrows():
        y = load_audio(row['archivo_origen'], row['tiempo_inicio'])
        if y is None or len(y) < SR_TARGET:
            continue
        freqs, psd_db = power_spectrum(y)
        psd_norm = normalize_psd(psd_db, freqs)
        all_psd.append(psd_norm)
        n_ok += 1

    if not all_psd:
        print(f"    ✗ Sin audio cargado para Cluster {cid}")
        continue

    all_psd = np.array(all_psd)   # (n_ok, n_freq)
    mean_psd = all_psd.mean(axis=0)
    std_psd  = all_psd.std(axis=0)

    cluster_spectra[cid] = {
        'freqs'  : freqs,
        'mean'   : mean_psd,
        'std'    : std_psd,
        'n'      : n_ok,
        'n_total': n_total,
    }
    print(f"    ✓ {n_ok} clips procesados")

# ── Preparar máscara de frecuencias para el gráfico ───────────────────────────
# Usamos las frecuencias del primer cluster disponible
ref_freqs = list(cluster_spectra.values())[0]['freqs']
freq_mask = (ref_freqs >= FMIN) & (ref_freqs <= FMAX)
plot_freqs = ref_freqs[freq_mask]

# ── Figura 2 × 2 ──────────────────────────────────────────────────────────────
print("\nGenerando figura de comparación espectral...")

# Layout: fila superior = 3 paneles individuales; fila inferior = overlay grande
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('white')

gs = fig.add_gridspec(
    2, 3,
    height_ratios=[1.0, 1.1],
    hspace=0.52,
    wspace=0.38,
    left=0.07, right=0.97,
    top=0.90, bottom=0.08,
)

panel_labels = ['A', 'B', 'C']

# ── Paneles individuales (fila 0) ─────────────────────────────────────────────
ax_list = []
for i, (cid, info) in enumerate(FROG_CLUSTERS.items()):
    ax = fig.add_subplot(gs[0, i])
    ax_list.append(ax)

    if cid not in cluster_spectra:
        ax.text(0.5, 0.5, 'Sin datos', transform=ax.transAxes,
                ha='center', va='center', color='grey')
        continue

    sp   = cluster_spectra[cid]
    freq = sp['freqs'][freq_mask]
    mean = sp['mean'][freq_mask]
    std  = sp['std'][freq_mask]
    color = info['color']

    # Banda de confianza ± 1 SD
    ax.fill_between(freq, mean - std, mean + std,
                    color=color, alpha=0.20, linewidth=0, label='±1 SD')
    # Espectro medio
    ax.plot(freq, mean, color=color, linewidth=1.8, label='Media')

    # Frecuencia dominante (en el rango de interés)
    dom_idx = np.argmax(mean)
    dom_f   = freq[dom_idx]
    ax.axvline(dom_f, color=color, linewidth=1.0, linestyle='--', alpha=0.7)
    ax.text(dom_f + 30, mean.max() - 3,
            f'{dom_f:.0f} Hz', fontsize=7.5, color=color, fontweight='bold')

    # Escala logarítmica en X (frecuencia)
    ax.set_xscale('log')
    ax.set_xlim(FMIN, FMAX)
    ax.set_xticks([200, 500, 1000, 2000, 4000, 8000])
    ax.set_xticklabels(['200', '500', '1k', '2k', '4k', '8k'], fontsize=8)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylim(-45, 3)
    ax.set_yticks([-40, -30, -20, -10, 0])
    ax.tick_params(labelsize=8)

    ax.set_xlabel('Frecuencia (Hz)', fontsize=9, labelpad=3)
    ax.set_ylabel('Potencia relativa (dB)', fontsize=9, labelpad=3) if i == 0 else None
    ax.set_title(
        f'{panel_labels[i]}  {info["label"]}  ·  n = {sp["n_total"]:,} segs',
        fontsize=10, fontweight='bold', color=color, pad=5, loc='left'
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, which='major', linestyle=':', linewidth=0.5, alpha=0.45, color='#aaaaaa')

    # Mini-leyenda
    ax.legend(fontsize=7.5, loc='lower right', frameon=True,
              framealpha=0.85, edgecolor='#cccccc')

# ── Panel overlay (fila 1, ocupa las 3 columnas) ──────────────────────────────
ax_ov = fig.add_subplot(gs[1, :])

for cid, info in FROG_CLUSTERS.items():
    if cid not in cluster_spectra:
        continue
    sp    = cluster_spectra[cid]
    freq  = sp['freqs'][freq_mask]
    mean  = sp['mean'][freq_mask]
    std   = sp['std'][freq_mask]
    color = info['color']
    ls    = info['linestyle']
    n_tot = sp['n_total']

    ax_ov.fill_between(freq, mean - std, mean + std,
                       color=color, alpha=0.12, linewidth=0)
    ax_ov.plot(freq, mean, color=color, linewidth=2.2,
               linestyle=ls, label=f'{info["label"]}  (n = {n_tot:,})')

    # Marcador de frecuencia dominante
    dom_idx = np.argmax(mean)
    dom_f   = freq[dom_idx]
    ax_ov.axvline(dom_f, color=color, linewidth=0.9,
                  linestyle=':', alpha=0.55)

ax_ov.set_xscale('log')
ax_ov.set_xlim(FMIN, FMAX)
ax_ov.set_xticks([200, 500, 1000, 2000, 4000, 8000])
ax_ov.set_xticklabels(['200', '500', '1k', '2k', '4k', '8k'], fontsize=9)
ax_ov.xaxis.set_minor_locator(mticker.NullLocator())
ax_ov.set_ylim(-45, 3)
ax_ov.set_yticks([-40, -30, -20, -10, 0])
ax_ov.tick_params(labelsize=9)
ax_ov.set_xlabel('Frecuencia (Hz)', fontsize=10, labelpad=4)
ax_ov.set_ylabel('Potencia relativa (dB)', fontsize=10, labelpad=4)
ax_ov.set_title(
    'D  Comparacion superpuesta — Tres clusters de vocalización de anfibio',
    fontsize=10.5, fontweight='bold', color='#222222', pad=5, loc='left'
)
ax_ov.spines['top'].set_visible(False)
ax_ov.spines['right'].set_visible(False)
ax_ov.grid(True, which='major', linestyle=':', linewidth=0.5, alpha=0.45, color='#aaaaaa')

ax_ov.legend(fontsize=9, loc='lower right', frameon=True,
             framealpha=0.90, edgecolor='#aaaaaa',
             title='Cluster  (segmentos totales)', title_fontsize=8.5)

# Anotación aclaratoria de la banda sombreada
ax_ov.text(0.01, 0.04,
           'Banda sombreada = ±1 desviación estándar entre segmentos del cluster.\n'
           'Las líneas verticales punteadas indican la frecuencia dominante de cada cluster.',
           transform=ax_ov.transAxes, fontsize=7.5,
           color='#777777', style='italic', va='bottom')

# ── Supertítulo ───────────────────────────────────────────────────────────────
fig.suptitle(
    'Huella espectral de los clusters de vocalización de anfibio — PA-17 (Pastizal) | Tapytá\n'
    f'Espectros de potencia medios calculados sobre muestras de hasta {N_SEG} segmentos de 4 s por cluster',
    fontsize=11.5, fontweight='bold', color='#1a1a2e', y=0.97
)

# ── Guardar ───────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'FigK_Comparacion_Espectral_Anfibios_PA17.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n  Figura guardada: {out_path}")

# ── Estadísticas espectrales por cluster ──────────────────────────────────────
print("\n" + "=" * 60)
print("ESTADÍSTICAS ESPECTRALES — Clusters de anfibio (PA-17)")
print("=" * 60)

for cid, info in FROG_CLUSTERS.items():
    if cid not in cluster_spectra:
        continue
    sp   = cluster_spectra[cid]
    freq = sp['freqs'][freq_mask]
    mean = sp['mean'][freq_mask]

    dom_idx = np.argmax(mean)
    dom_f   = freq[dom_idx]

    # Banda a -10 dB del pico (ancho de banda funcional)
    thresh = mean.max() - 10.0
    above  = freq[mean >= thresh]
    bw_lo  = above[0]  if len(above) > 0 else np.nan
    bw_hi  = above[-1] if len(above) > 0 else np.nan

    print(f"\n  {info['label']} (n = {sp['n_total']:,} segs, {sp['n']} clips):")
    print(f"    Freq. dominante : {dom_f:.0f} Hz")
    print(f"    Ancho de banda  : {bw_lo:.0f} – {bw_hi:.0f} Hz  (a -10 dB del pico)")
    print(f"    Pico (norm.)    : {mean.max():.1f} dB  (ref = max)")

print("\n" + "=" * 60)
print("Script K completado.")
