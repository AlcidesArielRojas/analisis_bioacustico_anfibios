# ================================================================
# Script 5: Espectrogramas emparejados + audios para validación
# (opción B: ventana más parecida según YAMNet)
# ------------------------------------------------
# DESCRIPCIÓN BREVE
#
# Este script:
#   - Lee la tabla de matching (Script 4).
#   - Para cada representativo toma el Top-1 match.
#   - Busca el .wav del representativo y el .wav de la base.
#   - Usa YAMNet para encontrar, en el representativo, la ventana
#     de 1.5 s cuya huella YAMNet es más parecida al embedding
#     de la base (Script 2).
#   - Genera:
#       - representativo completo,
#       - representativo recortado (ventana más parecida),
#       - base completa,
#       - base recortada (primeros 1.5 s),
#       - figura PNG con espectrogramas de los recortes,
#       - metadata.json con info del match.
# ================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import librosa.display

import matplotlib
matplotlib.use("Agg")  # backend sin interfaz gráfica
import matplotlib.pyplot as plt

import soundfile as sf
import json
import tensorflow as tf
import tensorflow_hub as hub

# ---------------- CONFIGURACIÓN ----------------

BASE_DIR = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local")

RUTA_MATCH = BASE_DIR / "matching_representativos_vs_base" / "matching_representativos_vs_base_topK.parquet"
RUTA_EMB_BASE = BASE_DIR / "embeddings_BD_anfibios" / "embeddings_BD_anfibios_yamnet.parquet"

NOMBRE_CAMPANIA = "Campaña diciembre 2024"
RUTA_REP_FASE3 = BASE_DIR / "figuras_inspeccion_clusters" / NOMBRE_CAMPANIA
RUTA_REP_FASE4_3 = BASE_DIR / "figuras_inspeccion_subclusters" / NOMBRE_CAMPANIA

RUTA_BD_AUDIO = BASE_DIR / "BD_anfibios_wav"

RUTA_SALIDA = BASE_DIR / "validacion_matching_rep_vs_base"
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

SR_OBJETIVO = 22050
SR_YAMNET = 16000
VENTANA_SEG = 1.5
N_MELS = 64


# ---------------- FUNCIONES AUXILIARES ----------------

def cargar_yamnet_model():
    print("Cargando modelo YAMNet desde TensorFlow Hub...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    return model


def buscar_representativo_wav(nombre_archivo: str) -> Path | None:
    candidatos = list(RUTA_REP_FASE3.rglob(nombre_archivo)) + \
                 list(RUTA_REP_FASE4_3.rglob(nombre_archivo))
    return candidatos[0] if candidatos else None


def buscar_bd_wav(nombre_archivo: str) -> Path | None:
    candidatos = list(RUTA_BD_AUDIO.rglob(nombre_archivo))
    return candidatos[0] if candidatos else None


def cargar_audio(ruta: Path, sr_objetivo: int = SR_OBJETIVO):
    y, sr = librosa.load(str(ruta), sr=sr_objetivo)
    return y, sr


def resample_audio(y, sr_orig, sr_target):
    if sr_orig == sr_target:
        return y
    return librosa.resample(y, orig_sr=sr_orig, target_sr=sr_target)


def embedding_yamnet_segment(y_seg, sr, yamnet_model):
    y16 = resample_audio(y_seg, sr, SR_YAMNET).astype(np.float32)
    if y16.ndim > 1:
        y16 = y16.mean(axis=1)
    scores, emb, _ = yamnet_model(y16)
    emb_np = emb.numpy()
    return emb_np.mean(axis=0)


def normalizar(v):
    n = np.linalg.norm(v) + 1e-12
    return v / n


def similitud_coseno(v1, v2):
    v1n = normalizar(v1)
    v2n = normalizar(v2)
    return float(np.dot(v1n, v2n))


def recorte_rep_por_similitud_yamnet(y_rep_full, sr_rep, base_emb, yamnet_model):
    n_total = len(y_rep_full)
    n_win = int(VENTANA_SEG * sr_rep)

    if n_total <= n_win:
        return y_rep_full, VENTANA_SEG if n_total >= sr_rep else n_total / sr_rep

    hop = int(0.5 * sr_rep)  # hop 0.5 s
    if hop <= 0:
        hop = 1

    best_sim = -np.inf
    best_seg = None

    for start in range(0, n_total - n_win + 1, hop):
        seg = y_rep_full[start:start + n_win]
        try:
            emb_seg = embedding_yamnet_segment(seg, sr_rep, yamnet_model)
        except Exception:
            continue
        sim = similitud_coseno(emb_seg, base_emb)
        if sim > best_sim:
            best_sim = sim
            best_seg = seg

    if best_seg is None:
        best_seg = y_rep_full[:n_win]

    return best_seg, VENTANA_SEG


def plot_mel(y, sr, ax, titulo: str):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024,
                                       hop_length=256, n_mels=N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(
        S_db, sr=sr, hop_length=256,
        x_axis="time", y_axis="mel",
        cmap="magma", ax=ax
    )
    ax.set_title(titulo, fontsize=9)


def generar_figura_pareada(y_rep, sr_rep, y_bd, sr_bd,
                           info_rep: dict, info_bd: dict, ruta_fig: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    titulo_rep = f"Representativo (ventana más parecida)\n{info_rep.get('rep_archivo', '')}"
    especie_txt = info_bd.get("base_especie", "") or ""
    titulo_bd = f"Base (candidata, primeros {VENTANA_SEG:.1f}s)\n{info_bd.get('base_archivo', '')}\n{especie_txt}"

    plot_mel(y_rep, sr_rep, axes[0], titulo_rep)
    plot_mel(y_bd, sr_bd, axes[1], titulo_bd)

    fig.suptitle("Comparación espectral: ventana más parecida según YAMNet", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    fig.savefig(ruta_fig)
    plt.close(fig)


def guardar_audio(ruta_destino: Path, y, sr):
    sf.write(str(ruta_destino), y, sr)


# ---------------- MAIN ----------------

def main():
    if not RUTA_MATCH.exists():
        print(f"⚠️ No se encontró el archivo de matching: {RUTA_MATCH}")
        return

    if not RUTA_EMB_BASE.exists():
        print(f"⚠️ No se encontró el archivo de embeddings de la base: {RUTA_EMB_BASE}")
        return

    df_match = pd.read_parquet(RUTA_MATCH)
    if df_match.empty:
        print("⚠️ La tabla de matching está vacía.")
        return

    df_base_emb = pd.read_parquet(RUTA_EMB_BASE)
    if df_base_emb.empty:
        print("⚠️ La tabla de embeddings de la base está vacía.")
        return

    cols_emb = sorted([c for c in df_base_emb.columns if c.startswith("emb_")],
                      key=lambda x: int(x.split("_")[1]))
    emb_dict = {}
    for _, row in df_base_emb.iterrows():
        nombre = row["archivo"]
        emb_vec = row[cols_emb].to_numpy(dtype=np.float32)
        emb_dict[nombre] = emb_vec

    df_top1 = df_match[df_match["rank"] == 1].copy()
    if df_top1.empty:
        print("⚠️ No hay filas con rank == 1 en el matching.")
        return

    print("\n===============================================")
    print("Generación de espectrogramas + audios (Top-1, ventana más parecida YAMNet)")
    print("Archivo matching:", RUTA_MATCH)
    print("Embeddings base :", RUTA_EMB_BASE)
    print("Salida (carpetas):", RUTA_SALIDA)
    print("Total de pares   :", len(df_top1))
    print("===============================================\n")

    yamnet_model = cargar_yamnet_model()

    n_ok = 0
    n_fail = 0

    for idx, row in df_top1.iterrows():
        rep_archivo = row.get("rep_archivo")
        base_archivo = row.get("base_archivo")

        if rep_archivo is None or base_archivo is None:
            print(f"⚠️ Fila sin nombres de archivo válidos (índice {idx}).")
            n_fail += 1
            continue

        if base_archivo not in emb_dict:
            print(f"⚠️ No se encontró embedding para el audio de base: {base_archivo}")
            n_fail += 1
            continue

        base_emb = emb_dict[base_archivo]

        ruta_rep = buscar_representativo_wav(rep_archivo)
        ruta_bd = buscar_bd_wav(base_archivo)

        if ruta_rep is None:
            print(f"⚠️ No se encontró el representativo: {rep_archivo}")
            n_fail += 1
            continue

        if ruta_bd is None:
            print(f"⚠️ No se encontró el audio de base: {base_archivo}")
            n_fail += 1
            continue

        nombre_par = f"{Path(rep_archivo).stem}__VS__{Path(base_archivo).stem}"
        carpeta_par = RUTA_SALIDA / nombre_par
        carpeta_par.mkdir(parents=True, exist_ok=True)

        try:
            y_rep_full, sr_rep = cargar_audio(ruta_rep)
            y_bd_full, sr_bd = cargar_audio(ruta_bd)
        except Exception as e:
            print(f"⚠️ Error al cargar audios para {rep_archivo} vs {base_archivo}: {e}")
            n_fail += 1
            continue

        y_rep_rec, dur_rec = recorte_rep_por_similitud_yamnet(
            y_rep_full, sr_rep, base_emb, yamnet_model
        )

        n_bd_win = int(VENTANA_SEG * sr_bd)
        if len(y_bd_full) <= n_bd_win:
            y_bd_rec = y_bd_full
            dur_bd_rec = len(y_bd_full) / sr_bd
        else:
            y_bd_rec = y_bd_full[:n_bd_win]
            dur_bd_rec = VENTANA_SEG

        ruta_rep_full = carpeta_par / "representativo_completo.wav"
        ruta_rep_rec = carpeta_par / "representativo_recortado.wav"
        ruta_bd_full = carpeta_par / "base_completa.wav"
        ruta_bd_rec = carpeta_par / "base_recortada.wav"

        try:
            guardar_audio(ruta_rep_full, y_rep_full, sr_rep)
            guardar_audio(ruta_rep_rec, y_rep_rec, sr_rep)
            guardar_audio(ruta_bd_full, y_bd_full, sr_bd)
            guardar_audio(ruta_bd_rec, y_bd_rec, sr_bd)
        except Exception as e:
            print(f"⚠️ Error al guardar audios para {rep_archivo} vs {base_archivo}: {e}")
            n_fail += 1
            continue

        ruta_fig = carpeta_par / "espectrograma_rep_vs_base.png"
        info_rep = {
            "rep_archivo": rep_archivo,
            "rep_sitio": row.get("rep_sitio"),
            "rep_grupo": row.get("rep_grupo"),
        }
        info_bd = {
            "base_archivo": base_archivo,
            "base_especie": row.get("base_especie"),
        }

        try:
            generar_figura_pareada(
                y_rep=y_rep_rec,
                sr_rep=sr_rep,
                y_bd=y_bd_rec,
                sr_bd=sr_bd,
                info_rep=info_rep,
                info_bd=info_bd,
                ruta_fig=ruta_fig
            )
        except Exception as e:
            print(f"⚠️ Error al generar figura para {rep_archivo} vs {base_archivo}: {e}")
            n_fail += 1
            continue

        metadata = {
            "rep_archivo": rep_archivo,
            "rep_sitio": row.get("rep_sitio"),
            "rep_grupo": row.get("rep_grupo"),
            "base_archivo": base_archivo,
            "base_especie": row.get("base_especie"),
            "duracion_rep_rec_s": float(dur_rec),
            "duracion_base_rec_s": float(dur_bd_rec),
            "similitud_coseno_match": float(row.get("similitud_coseno", np.nan)),
        }
        ruta_meta = carpeta_par / "metadata.json"
        with open(ruta_meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"✓ Carpeta generada: {carpeta_par.name}")
        n_ok += 1

    print("\nResumen:")
    print(f"  Pares generados correctamente: {n_ok}")
    print(f"  Fallos (archivos no encontrados / errores): {n_fail}")


if __name__ == "__main__":
    main()