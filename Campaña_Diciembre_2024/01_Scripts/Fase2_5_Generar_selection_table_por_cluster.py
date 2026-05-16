# ================================================================
# Fase 2.5 por campaña
# Partir la Selection Table global en carpetas por sitio y cluster
# ------------------------------------------------
# ¿Qué hace este script?
#   - Para cada sitio (grabadora) de la campaña:
#       * Carga la Selection Table global generada en Fase 2
#         (un archivo por sitio con todos los clusters).
#       * Verifica que exista la columna 'Annotation', donde se
#         guarda el nombre del cluster (p. ej. "cluster_0").
#       * Filtra filas sin anotación válida.
#       * Separa la tabla en varias tablas, una por cada valor
#         de 'Annotation' (es decir, una por cluster).
#       * Reenumera la columna 'Selection' dentro de cada cluster.
#       * Limpia y ordena las columnas (tiempos, frecuencias,
#         Annotation, archivo_origen, sitio, etc.).
#       * Guarda cada Selection Table de cluster en una carpeta
#         propia: BASE/sitio/cluster_X/...
#
# En resumen: toma la Selection Table global de Fase 2 y la
# reorganiza en archivos más pequeños, uno por cluster, listos
# para trabajar en Raven por sitio y cluster.
# ================================================================

import pandas as pd
from pathlib import Path

# ================================================================
# Fase 2.5: Partir Selection Table global por cluster (por campaña)
# ================================================================

NOMBRE_CAMPANIA = "Campaña diciembre 2024"
SUFIJO_CORRIDA = "v2_horario18a06_insectos6a12"

BASE_DIR = Path(r"C:/Users/User/Proyecto_Paisajes_Sonoros_Repositorio_Local")
BASE_RESULTADOS = BASE_DIR / "resultados_HDD_Seagate" / NOMBRE_CAMPANIA

# Carpeta donde se guardarán las selection tables por sitio y cluster
RUTA_SALIDA = BASE_DIR / "selection_tables_por_cluster_desde_raven" / NOMBRE_CAMPANIA
RUTA_SALIDA.mkdir(exist_ok=True, parents=True)

def split_selection_table_por_cluster(sitio: str, ruta_selection_table: Path):
    print(f"\n==============================================")
    print(f"Partiendo Selection Table global por cluster: {sitio}")
    print(f"Archivo: {ruta_selection_table}")
    print(f"==============================================")

    if not ruta_selection_table.exists():
        print(f"⚠️ No existe Selection Table para {sitio}: {ruta_selection_table}")
        return

    df = pd.read_csv(ruta_selection_table, sep="\t")

    if "Annotation" not in df.columns:
        print(f"⚠️ {sitio}: la Selection Table no tiene columna 'Annotation'. Columnas: {list(df.columns)}")
        return

    df = df.dropna(subset=["Annotation"])
    df = df[df["Annotation"].astype(str).str.strip() != ""]
    if df.empty:
        print(f"⚠️ {sitio}: no hay filas con Annotation válida.")
        return

    clusters = sorted(df["Annotation"].astype(str).unique())
    print(f"{sitio}: {len(clusters)} clusters encontrados en Annotation: {clusters}")

    carpeta_sitio = RUTA_SALIDA / sitio
    carpeta_sitio.mkdir(exist_ok=True)

    for cl in clusters:
        df_c = df[df["Annotation"].astype(str) == cl].copy()
        if df_c.empty:
            continue

        if "Selection" in df_c.columns:
            df_c = df_c.reset_index(drop=True)
            df_c["Selection"] = df_c.index + 1

        for col in ["Begin Time (s)", "End Time (s)"]:
            if col in df_c.columns:
                df_c[col] = pd.to_numeric(df_c[col], errors="coerce")

        if {"Begin Time (s)", "End Time (s)"}.issubset(df_c.columns):
            df_c = df_c.dropna(subset=["Begin Time (s)", "End Time (s)"])
            if df_c.empty:
                print(f"⚠️ {sitio} | {cl}: todas las filas tienen tiempos inválidos.")
                continue

        if "archivo_origen" in df_c.columns:
            df_c["Annotation"] = (
                df_c["Annotation"].astype(str)
                + " | "
                + df_c["archivo_origen"].astype(str)
            )

        columnas_base = [
            "Selection", "View", "Channel",
            "Begin Time (s)", "End Time (s)",
            "Low Freq (Hz)", "High Freq (Hz)",
            "Annotation"
        ]
        extras = []
        for extra_col in ["archivo_origen", "sitio"]:
            if extra_col in df_c.columns:
                extras.append(extra_col)

        otras = [c for c in df_c.columns if c not in columnas_base + extras]
        columnas_finales = [c for c in columnas_base if c in df_c.columns] + extras + otras
        df_c = df_c[columnas_finales]

        carpeta_cluster = carpeta_sitio / cl
        carpeta_cluster.mkdir(exist_ok=True)

        out_path = carpeta_cluster / f"{sitio}_{cl}_selection_table.txt"
        df_c.to_csv(out_path, sep="\t", index=False)

        print(f"   ✓ {sitio} | {cl}: {len(df_c)} selecciones → {out_path}")

    print(f"✔️ Listo: Selection Tables por cluster generadas para {sitio}")


if __name__ == "__main__":
    sitios = [p.name for p in BASE_RESULTADOS.iterdir() if p.is_dir()]
    sitios = sorted(sitios)

    print("\nSitios detectados para Fase 2.5:")
    for s in sitios:
        print(" -", s)

    for sitio in sitios:
        ruta_sitio = BASE_RESULTADOS / sitio
        ruta_st = ruta_sitio / f"{sitio}_{SUFIJO_CORRIDA}_fase2_raven_selection_table.txt"
        split_selection_table_por_cluster(sitio, ruta_st)

    print("\n🎉 Proceso completado. Las Selection Tables por cluster están listas para Raven Lite.")