# ================================================================
# Script 4: Matching automático entre representativos y base de especies
# ------------------------------------------------
# DESCRIPCIÓN BREVE
#
# Este script:
#   - Carga los embeddings de la base de especies (Script 2)
#     y los embeddings de los audios representativos (Script 3).
#   - Calcula la similitud (coseno) entre cada representativo
#     y todos los audios de la base.
#   - Para cada representativo, guarda los Top-K matches
#     (los más parecidos) en una tabla.
#
# No genera audios ni figuras: solo una tabla de resultados
# que luego se puede usar para validación auditiva y visual.
# ================================================================

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------- CONFIGURACIÓN ----------------

BASE_DIR = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local")

# Embeddings de la base de especies (salida del Script 2)
RUTA_EMB_BASE = BASE_DIR / "embeddings_BD_anfibios" / "embeddings_BD_anfibios_yamnet.parquet"

# Embeddings de los audios representativos (salida del Script 3)
RUTA_EMB_REP = BASE_DIR / "embeddings_representativos" / "embeddings_representativos_yamnet.parquet"

# Carpeta de salida para la tabla de matches
RUTA_SALIDA = BASE_DIR / "matching_representativos_vs_base"
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

# Nombre del archivo de salida
ARCHIVO_SALIDA = RUTA_SALIDA / "matching_representativos_vs_base_topK.parquet"

# Número de mejores coincidencias a guardar por representativo
TOP_K = 5


# ---------------- FUNCIONES AUXILIARES ----------------

def extraer_matriz_embeddings(df: pd.DataFrame, prefijo: str = "emb_") -> tuple[np.ndarray, list[str]]:
    """
    Dado un DataFrame con columnas emb_0, emb_1, ..., devuelve
    una matriz numpy (n_filas, n_dim) y la lista de columnas usadas.
    """
    cols_emb = sorted(
        [c for c in df.columns if c.startswith(prefijo)],
        key=lambda x: int(x.split("_")[1])
    )
    if not cols_emb:
        raise ValueError("No se encontraron columnas de embeddings con el prefijo especificado.")
    return df[cols_emb].to_numpy(dtype=np.float32), cols_emb


def normalizar_filas(X: np.ndarray) -> np.ndarray:
    """
    Normaliza cada fila de X a norma 1 (para similitud coseno).
    """
    normas = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / normas


def calcular_topK_similitud_coseno(
    X_rep: np.ndarray,
    X_base: np.ndarray,
    df_rep: pd.DataFrame,
    df_base: pd.DataFrame,
    top_k: int = TOP_K
) -> pd.DataFrame:
    """
    Calcula similitud coseno entre cada representativo y todos los
    elementos de la base, y devuelve un DataFrame con los Top-K
    matches por representativo.
    """
    # Normalizar para similitud coseno
    X_rep_norm = normalizar_filas(X_rep)
    X_base_norm = normalizar_filas(X_base)

    # Producto punto = similitud coseno (porque están normalizados)
    sim = X_rep_norm @ X_base_norm.T  # (N_rep, N_base)

    registros = []

    for i_rep in range(X_rep.shape[0]):
        sim_i = sim[i_rep]

        # índices de los Top-K más altos
        if top_k >= len(sim_i):
            idx_top = np.argsort(-sim_i)
        else:
            idx_top = np.argpartition(-sim_i, top_k)[:top_k]
            idx_top = idx_top[np.argsort(-sim_i[idx_top])]

        fila_rep = df_rep.iloc[i_rep]

        for rank, j_base in enumerate(idx_top, start=1):
            fila_base = df_base.iloc[j_base]
            sim_val = float(sim_i[j_base])

            registros.append({
                # Info del representativo (Script 3)
                "rep_sitio": fila_rep.get("sitio", None),
                "rep_grupo": fila_rep.get("grupo", None),
                "rep_archivo": fila_rep.get("archivo", None),
                "rep_ruta_relativa_desde_BASE_DIR": fila_rep.get("ruta_relativa_desde_BASE_DIR", None),
                "rep_duracion_s": fila_rep.get("duracion_s", None),
                "rep_n_ventanas": fila_rep.get("n_ventanas", None),

                # Info del elemento en la base (Script 2)
                "base_especie": fila_base.get("especie", None),
                "base_archivo": fila_base.get("archivo", None),
                "base_ruta_relativa": fila_base.get("ruta_relativa", None),
                "base_duracion_s": fila_base.get("duracion_s", None),
                "base_n_ventanas": fila_base.get("n_ventanas", None),

                # Métrica de similitud
                "rank": rank,
                "similitud_coseno": sim_val,
            })

    return pd.DataFrame(registros)


# ---------------- MAIN ----------------

def main():
    if not RUTA_EMB_BASE.exists():
        print(f"⚠️ No se encontró el archivo de embeddings de la base: {RUTA_EMB_BASE}")
        return

    if not RUTA_EMB_REP.exists():
        print(f"⚠️ No se encontró el archivo de embeddings de representativos: {RUTA_EMB_REP}")
        return

    print("\n===============================================")
    print("Matching automático entre representativos y base de especies")
    print("Archivo base        :", RUTA_EMB_BASE)
    print("Archivo representat.:", RUTA_EMB_REP)
    print("Salida              :", ARCHIVO_SALIDA)
    print("TOP-K               :", TOP_K)
    print("===============================================\n")

    # Cargar tablas
    df_base = pd.read_parquet(RUTA_EMB_BASE)
    df_rep = pd.read_parquet(RUTA_EMB_REP)

    if df_base.empty:
        print("⚠️ La tabla de embeddings de la base está vacía.")
        return

    if df_rep.empty:
        print("⚠️ La tabla de embeddings de representativos está vacía.")
        return

    # Extraer matrices de embeddings
    X_base, cols_base = extraer_matriz_embeddings(df_base, prefijo="emb_")
    X_rep, cols_rep = extraer_matriz_embeddings(df_rep, prefijo="emb_")

    if X_base.shape[1] != X_rep.shape[1]:
        print(f"⚠️ Dimensión de embeddings incompatible: base={X_base.shape[1]}, rep={X_rep.shape[1]}")
        return

    print(f"Embeddings base        : {X_base.shape[0]} registros, dim={X_base.shape[1]}")
    print(f"Embeddings representat.: {X_rep.shape[0]} registros, dim={X_rep.shape[1]}")

    # Calcular Top-K matches
    df_matches = calcular_topK_similitud_coseno(
        X_rep=X_rep,
        X_base=X_base,
        df_rep=df_rep,
        df_base=df_base,
        top_k=TOP_K
    )

    if df_matches.empty:
        print("⚠️ No se generaron matches.")
        return

    df_matches.to_parquet(ARCHIVO_SALIDA, index=False)
    print("\n✔ Matching completado.")
    print("  Archivo guardado en:", ARCHIVO_SALIDA)
    print("  Total de filas (rep × Top-K):", len(df_matches))


if __name__ == "__main__":
    main()