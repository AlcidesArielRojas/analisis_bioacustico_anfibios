# ================================================================
# Script 2 NUEVO: Proyección de audios BD en UMAP real
# ------------------------------------------------
# - Carga modelos_directos_{SITE}_{SUFIJO_CORRIDA}.pkl
# - Carga MFCC de la BD por sitio
# - Aplica scaler -> PCA -> UMAP.transform
# - Asigna clusters/subclusters (si ya tenés esa info)
# - Guarda CSV con coordenadas UMAP reales de la BD
# ================================================================

from pathlib import Path
import pandas as pd
import joblib
import numpy as np

NOMBRE_CAMPANIA = "Campaña diciembre 2024"
SUFIJO_CORRIDA = "v2_horario18a06_insectos6a12"

BASE_DIR = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local")
BASE_RESULTADOS = BASE_DIR / "resultados_HDD_Seagate" / NOMBRE_CAMPANIA
RUTA_MODELOS = BASE_DIR / "modelos_fase2_fase4" / NOMBRE_CAMPANIA

# Carpeta donde guardar resultados de proyección BD
RUTA_PROYECCION_BD = BASE_DIR / "proyeccion_BD_fase1_5"
RUTA_PROYECCION_BD.mkdir(parents=True, exist_ok=True)


def cargar_features_bd(sitio: str) -> pd.DataFrame:
    """
    Cargar MFCC de la BD (archivo único global).
    """
    ruta_bd = RUTA_PROYECCION_BD / "features_BD_anfibios.parquet"
    if not ruta_bd.exists():
        raise FileNotFoundError(f"No se encontró archivo global de MFCC BD: {ruta_bd}")
    return pd.read_parquet(ruta_bd)



def proyectar_bd_en_umap_real(sitio: str):
    print(f"\n=== Proyección BD en UMAP real para sitio: {sitio} ===")

    ruta_modelo = RUTA_MODELOS / f"modelos_directos_{sitio}_{SUFIJO_CORRIDA}.pkl"
    if not ruta_modelo.exists():
        print(f"⚠️ No se encontró modelo directo para {sitio}: {ruta_modelo}")
        return

    modelo = joblib.load(ruta_modelo)
    scaler = modelo["scaler"]
    pca = modelo["pca"]
    umap_model = modelo["umap_model"]
    features = modelo["features"]

    # Cargar MFCC de BD
    try:
        df_bd = cargar_features_bd(sitio)
    except FileNotFoundError as e:
        print(e)
        return

    # Asegurar que las columnas de features existan
    missing = [c for c in features if c not in df_bd.columns]
    if missing:
        print(f"⚠️ Faltan columnas de features en BD para {sitio}: {missing[:5]} ...")
        return

    X_bd = df_bd[features].dropna()
    idx = X_bd.index

    if len(X_bd) == 0:
        print(f"⚠️ No hay filas válidas de BD para {sitio}")
        return

    # Proyección: scaler -> PCA -> UMAP
    X_bd_scaled = scaler.transform(X_bd.values)
    X_bd_pca = pca.transform(X_bd_scaled)
    X_bd_umap = umap_model.transform(X_bd_pca)

    df_bd_umap = df_bd.loc[idx].copy()
    df_bd_umap["BD_DIM1"] = X_bd_umap[:, 0]
    df_bd_umap["BD_DIM2"] = X_bd_umap[:, 1]
    df_bd_umap["BD_U1"] = X_bd_umap[:, 0]
    df_bd_umap["BD_U2"] = X_bd_umap[:, 1]
    df_bd_umap["BD_U3"] = X_bd_umap[:, 2] if X_bd_umap.shape[1] >= 3 else 0.0

    # Si ya tenés matching a clusters/subclusters en otro archivo, acá podrías mergear.
    # Por ejemplo:
    # ruta_matching = RUTA_PROYECCION_BD / f"matching_BD_clusters_{sitio}_{SUFIJO_CORRIDA}.csv"
    # df_match = pd.read_csv(ruta_matching)
    # df_bd_umap = df_bd_umap.merge(df_match, on=["archivo_origen", "tiempo_inicio", "tiempo_fin"], how="left")

    ruta_salida = RUTA_PROYECCION_BD / f"{sitio}_{SUFIJO_CORRIDA}_proyeccion_BD_umap_real.csv"
    df_bd_umap.to_csv(ruta_salida, index=False)
    print(f"💾 Proyección BD UMAP real guardada en: {ruta_salida}")


def main():
    if not RUTA_MODELOS.exists():
        print(f"⚠️ Carpeta de modelos no existe: {RUTA_MODELOS}")
        return

    # Detectar sitios a partir de modelos directos
    modelos = sorted([
        p.name for p in RUTA_MODELOS.iterdir()
        if p.is_file() and p.name.startswith("modelos_directos_") and p.name.endswith(f"_{SUFIJO_CORRIDA}.pkl")
    ])

    if not modelos:
        print("⚠️ No se encontraron modelos directos. ¿Corriste Fase 2 corregida?")
        return

    sitios = sorted({fname.split("modelos_directos_")[1].split(f"_{SUFIJO_CORRIDA}.pkl")[0]
                     for fname in modelos})

    print("Sitios detectados para proyección BD:")
    for s in sitios:
        print(" -", s)

    for sitio in sitios:
        proyectar_bd_en_umap_real(sitio)

    print("\n✅ Proyección BD completada.")


if __name__ == "__main__":
    main()
