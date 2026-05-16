# ================================================================
# Fase 4.3: Visualización automática de subclusters (extendida)
# ------------------------------------------------
# DESCRIPCIÓN BREVE
#
# Este script toma los resultados de Fase 4.2 y genera visualizaciones
# y audios para inspeccionar cómo son los sonidos dentro de cada
# subcluster. Produce:
#   - UMAP 2D y 3D por subcluster,
#   - comparación cluster original vs subclusters,
#   - histogramas de distancias al centroide,
#   - grillas de espectrogramas (representativos, aleatorios, extremos),
#   - audios WAV de los segmentos representativos,
#   - un PDF resumen por sitio.
#
# No modifica datos: solo genera figuras y audios para análisis.
# ================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin interfaz gráfica
import matplotlib.pyplot as plt
import librosa
import librosa.display
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px
import soundfile as sf

# ---------------- CONFIGURACIÓN ----------------

NOMBRE_CAMPANIA = "Campaña diciembre 2024"
SUFIJO_CORRIDA = "v2_horario18a06_insectos6a12"

BASE_DIR = Path(r"C:/Users/User/Proyecto_Paisajes_Sonoros_Repositorio_Local")
BASE_RESULTADOS = BASE_DIR / "resultados_HDD_Seagate" / NOMBRE_CAMPANIA

AUDIO_ROOT = Path(r"D:/") / NOMBRE_CAMPANIA

N_MUESTRAS_ALEATORIAS = 10
N_MUESTRAS_EXTREMOS = 10

SR_OBJETIVO = 22050

RUTA_FIGURAS = BASE_DIR / "figuras_inspeccion_subclusters" / NOMBRE_CAMPANIA
RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)

plt.rcParams["savefig.dpi"] = 150


# ---------------- FUNCIONES DE AUDIO ----------------

def cargar_segmento_audio_desde_umap(row, audio_root, sr_objetivo=SR_OBJETIVO):
    archivo_rel = row["archivo_origen"]
    sitio = archivo_rel.split("/")[0]
    nombre_audio = Path(archivo_rel).name

    ruta_audio = audio_root / sitio / "Data" / nombre_audio

    t_ini = float(row["tiempo_inicio"])
    t_fin = float(row["tiempo_fin"])

    if not ruta_audio.exists():
        raise FileNotFoundError(f"No se encontró el audio: {ruta_audio}")

    y, sr = librosa.load(
        ruta_audio,
        sr=sr_objetivo,
        offset=t_ini,
        duration=(t_fin - t_ini)
    )
    return y, sr, ruta_audio


def guardar_audio_segmento(row, sitio, nombre_sufijo):
    try:
        y, sr, ruta_audio = cargar_segmento_audio_desde_umap(row, AUDIO_ROOT)
    except FileNotFoundError:
        print(f"⚠️ No se pudo guardar audio, archivo no encontrado: {row['archivo_origen']}")
        return

    carpeta = RUTA_FIGURAS / sitio / "audios_representativos"
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta_wav = carpeta / f"{sitio}_{nombre_sufijo}.wav"
    sf.write(ruta_wav, y, sr)
    print(f"🎧 Audio guardado: {ruta_wav}")


# ---------------- ESPECTROGRAMAS ----------------

def plot_mel_segmento(y, sr, ax, titulo=None):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024,
                                       hop_length=256, n_mels=64)
    S_db = librosa.power_to_db(S, ref=np.max)

    librosa.display.specshow(
        S_db, sr=sr, hop_length=256,
        x_axis="time", y_axis="mel",
        cmap="magma", ax=ax
    )

    ax.set_aspect("auto")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_anchor("C")

    if titulo:
        ax.set_title(titulo, fontsize=7)


# ---------------- SELECCIÓN DE SEGMENTOS ----------------

def seleccionar_representativos_aleatorios_extremos(df_sub):
    coords = df_sub[["U1", "U2", "U3"]].values
    centroide = coords.mean(axis=0)
    dist = np.linalg.norm(coords - centroide, axis=1)

    df_sub = df_sub.copy()
    df_sub["dist_centroide"] = dist

    df_rep = df_sub.nsmallest(2, "dist_centroide")
    df_rand = df_sub.sample(n=min(N_MUESTRAS_ALEATORIAS, len(df_sub)), random_state=42)
    df_ext = df_sub.nlargest(min(N_MUESTRAS_EXTREMOS, len(df_sub)), "dist_centroide")

    return df_rep, df_rand, df_ext


# ---------------- GRILLAS ----------------

def _titulo_compacto(row, ruta_audio, sub_id):
    nombre_corto = Path(ruta_audio).stem[-6:]  # últimos 6 caracteres del nombre
    return f"sub{sub_id} | {nombre_corto} | {row['tiempo_inicio']:.1f}-{row['tiempo_fin']:.1f}s"


def graficar_grilla(df_seg, titulo_fig, sitio, nombre_sufijo):
    if df_seg.empty:
        print(f"⚠️ Grilla vacía para {titulo_fig}")
        return None

    n = len(df_seg)
    n_cols = min(5, n)
    n_rows = int(np.ceil(n / n_cols))

    # Ajustar tamaño según cantidad de segmentos
    if n <= 2:
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 3))
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))

    axes = np.array(axes).reshape(n_rows, n_cols)

    for i, (_, row) in enumerate(df_seg.iterrows()):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]

        try:
            y, sr, ruta_audio = cargar_segmento_audio_desde_umap(row, AUDIO_ROOT)
        except FileNotFoundError:
            ax.set_title("Audio no encontrado", fontsize=7)
            ax.axis("off")
            continue

        sub_id = int(row["subcluster_id"])
        titulo = _titulo_compacto(row, ruta_audio, sub_id)
        plot_mel_segmento(y, sr, ax, titulo=titulo)

    # Apagar celdas vacías
    for j in range(n, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    fig.suptitle(titulo_fig, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.subplots_adjust(wspace=0.25, hspace=0.35)

    carpeta = RUTA_FIGURAS / sitio
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_fig = carpeta / f"{sitio}_{nombre_sufijo}.png"
    fig.savefig(ruta_fig)
    plt.close(fig)

    print(f"✓ Figura guardada: {ruta_fig}")
    return ruta_fig


# ---------------- VISUALIZACIONES UMAP ----------------

def generar_umap_2d_por_subcluster(df_valid, sitio, pdf=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df_valid["U1"], df_valid["U2"],
                    c=df_valid["subcluster_id"], cmap="tab20",
                    s=5, alpha=0.7)
    ax.set_title(f"{sitio} | UMAP 2D por subcluster")
    ax.set_xlabel("U1")
    ax.set_ylabel("U2")
    plt.colorbar(sc, ax=ax, label="subcluster_id")
    plt.tight_layout()

    ruta = RUTA_FIGURAS / sitio / f"{sitio}_umap2d_subclusters.png"
    fig.savefig(ruta)
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)


def generar_umap_2d_cluster_vs_subcluster(df_valid, sitio, pdf=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc1 = axes[0].scatter(df_valid["U1"], df_valid["U2"],
                          c=df_valid["cluster_hdbscan"], cmap="tab20",
                          s=5, alpha=0.7)
    axes[0].set_title("UMAP por cluster_hdbscan")
    axes[0].set_xlabel("U1")
    axes[0].set_ylabel("U2")
    plt.colorbar(sc1, ax=axes[0], label="cluster_hdbscan")

    sc2 = axes[1].scatter(df_valid["U1"], df_valid["U2"],
                          c=df_valid["subcluster_id"], cmap="tab20",
                          s=5, alpha=0.7)
    axes[1].set_title("UMAP por subcluster_id")
    axes[1].set_xlabel("U1")
    axes[1].set_ylabel("U2")
    plt.colorbar(sc2, ax=axes[1], label="subcluster_id")

    fig.suptitle(f"{sitio} | Comparación cluster vs subcluster", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    ruta = RUTA_FIGURAS / sitio / f"{sitio}_umap2d_cluster_vs_subcluster.png"
    fig.savefig(ruta)
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)


def generar_umap_3d_por_subcluster(df_valid, sitio):
    fig3d = px.scatter_3d(
        df_valid, x="U1", y="U2", z="U3",
        color="subcluster_id",
        hover_data=["cluster_hdbscan", "archivo_origen", "tiempo_inicio", "tiempo_fin"],
        title=f"{sitio} | UMAP 3D por subcluster"
    )
    fig3d.update_traces(marker=dict(size=3, opacity=0.8))
    fig3d.update_layout(width=800, height=600)

    ruta = RUTA_FIGURAS / sitio / f"{sitio}_umap3d_subclusters.html"
    fig3d.write_html(str(ruta))


def generar_histogramas_distancias(df_valid, sitio, pdf=None):
    df = df_valid.copy()
    dist_list = []

    for keys, df_sub in df.groupby(["cluster_hdbscan", "subcluster_id"]):
        coords = df_sub[["U1", "U2", "U3"]].values
        centroide = coords.mean(axis=0)
        dist = np.linalg.norm(coords - centroide, axis=1)
        df.loc[df_sub.index, "dist_centroide"] = dist
        dist_list.append((keys[0], keys[1], dist))

    fig, ax = plt.subplots(figsize=(8, 6))
    for k_cluster, sub_id, dist in dist_list:
        ax.hist(dist, bins=30, alpha=0.4, label=f"cl={k_cluster}, sub={sub_id}")

    ax.set_title(f"{sitio} | Distancias al centroide")
    ax.set_xlabel("distancia")
    ax.set_ylabel("frecuencia")
    ax.legend(fontsize=8)
    plt.tight_layout()

    ruta = RUTA_FIGURAS / sitio / f"{sitio}_hist_distancias.png"
    fig.savefig(ruta)
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)


# ---------------- PROCESO PRINCIPAL ----------------

def procesar_sitio(sitio):
    ruta_csv = BASE_RESULTADOS / sitio / f"{sitio}_{SUFIJO_CORRIDA}_fase4_umap_hdbscan_subclusters.csv"
    if not ruta_csv.exists():
        print(f"⚠️ No existe CSV de Fase 4.2 para {sitio}")
        return

    df = pd.read_csv(ruta_csv)

    df_valid = df[(df["cluster_hdbscan"] != -1) & (df["subcluster_id"] >= 0)].copy()
    if df_valid.empty:
        print(f"⚠️ {sitio}: no hay subclusters válidos")
        return

    carpeta = RUTA_FIGURAS / sitio
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_pdf = carpeta / f"{sitio}_resumen_subclusters.pdf"

    with PdfPages(ruta_pdf) as pdf:

        generar_umap_2d_por_subcluster(df_valid, sitio, pdf)
        generar_umap_2d_cluster_vs_subcluster(df_valid, sitio, pdf)
        generar_histogramas_distancias(df_valid, sitio, pdf)
        generar_umap_3d_por_subcluster(df_valid, sitio)

        grupos = df_valid.groupby(["cluster_hdbscan", "subcluster_id"])

        for keys, df_sub in grupos:
            k_cluster, sub_id = keys

            print(f"\nSitio={sitio} | cluster={k_cluster} | subcluster={sub_id} | n={len(df_sub)}")

            df_rep, df_rand, df_ext = seleccionar_representativos_aleatorios_extremos(df_sub)

            titulo_base = f"{sitio} | cl={k_cluster} | sub={sub_id}"

            # Guardar audios representativos
            for idx, (_, row_rep) in enumerate(df_rep.iterrows()):
                guardar_audio_segmento(
                    row_rep,
                    sitio,
                    nombre_sufijo=f"cl{k_cluster}_sub{sub_id}_representativo{idx+1}"
                )

            # Grillas PNG
            graficar_grilla(
                df_rep,
                f"{titulo_base} | Representativos",
                sitio,
                f"cl{k_cluster}_sub{sub_id}_representativos"
            )

            graficar_grilla(
                df_rand,
                f"{titulo_base} | Aleatorios",
                sitio,
                f"cl{k_cluster}_sub{sub_id}_aleatorios"
            )

            graficar_grilla(
                df_ext,
                f"{titulo_base} | Extremos",
                sitio,
                f"cl{k_cluster}_sub{sub_id}_extremos"
            )

            # Versión compacta para PDF (solo representativos)
            if not df_rep.empty:
                n = len(df_rep)
                n_cols = n
                n_rows = 1
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n, 3))
                axes = np.array(axes).reshape(n_rows, n_cols)

                for i, (_, row) in enumerate(df_rep.iterrows()):
                    ax = axes[0, i]
                    try:
                        y, sr, ruta_audio = cargar_segmento_audio_desde_umap(row, AUDIO_ROOT)
                    except FileNotFoundError:
                        ax.set_title("Audio no encontrado", fontsize=7)
                        ax.axis("off")
                        continue

                    titulo = _titulo_compacto(row, ruta_audio, sub_id)
                    plot_mel_segmento(y, sr, ax, titulo)

                fig.suptitle(f"{titulo_base} | Representativos (PDF)", fontsize=10)
                plt.tight_layout(rect=[0, 0, 1, 0.94])
                plt.subplots_adjust(wspace=0.25, hspace=0.35)
                pdf.savefig(fig)
                plt.close(fig)

    print(f"📄 PDF resumen guardado: {ruta_pdf}")


# ---------------- MAIN ----------------

def main():
    if not BASE_RESULTADOS.exists():
        print("⚠️ Carpeta de resultados no encontrada")
        return

    sitios = sorted([p.name for p in BASE_RESULTADOS.iterdir() if p.is_dir()])

    print("\nSitios detectados:")
    for s in sitios:
        print(" -", s)

    for sitio in sitios:
        print(f"\n==============================")
        print(f"Procesando sitio: {sitio}")
        print(f"==============================")
        procesar_sitio(sitio)


if __name__ == "__main__":
    main()