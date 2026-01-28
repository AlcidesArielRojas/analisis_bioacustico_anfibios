import pandas as pd
from pathlib import Path

# ================================================================
# CONFIGURACIÓN
# ================================================================

BASE_DIR = Path(r"C:/Users/User/Proyecto_Paisajes_Sonoros_Repositorio_Local")
RUTA_RESULTADOS = BASE_DIR / "resultados"

# Selection Tables globales (las que generaste en Fase 2)
SELECTION_TABLES = {
    "BO-Tapyta": RUTA_RESULTADOS / "BO-Tapyta_v2_horario18a06_insectos6a12_fase2_raven_selection_table.txt",
    "CANTERA1Tapy": RUTA_RESULTADOS / "CANTERA1Tapy_v2_horario18a06_insectos6a12_fase2_raven_selection_table.txt",
    "EU-41Tapyta": RUTA_RESULTADOS / "EU-41Tapyta_v2_horario18a06_insectos6a12_fase2_raven_selection_table.txt",
    "PA-41Tapyta": RUTA_RESULTADOS / "PA-41Tapyta_v2_horario18a06_insectos6a12_fase2_raven_selection_table.txt",
}

# Carpeta donde se guardarán las selection tables por sitio y cluster
RUTA_SALIDA = BASE_DIR / "selection_tables_por_cluster_desde_raven"
RUTA_SALIDA.mkdir(exist_ok=True, parents=True)

# ================================================================
# FUNCIÓN PRINCIPAL
# ================================================================

def split_selection_table_por_cluster(sitio: str, ruta_selection_table: Path):
    print(f"\n==============================================")
    print(f"Partiendo Selection Table global por cluster: {sitio}")
    print(f"Archivo: {ruta_selection_table}")
    print(f"==============================================")

    if not ruta_selection_table.exists():
        print(f"⚠️ No existe Selection Table para {sitio}: {ruta_selection_table}")
        return

    # Leer Selection Table de Raven (tabulado)
    df = pd.read_csv(ruta_selection_table, sep="\t")

    # Verificar columna de cluster
    if "Annotation" not in df.columns:
        print(f"⚠️ {sitio}: la Selection Table no tiene columna 'Annotation'. Columnas disponibles: {list(df.columns)}")
        return

    # Filtrar filas que tengan algo en Annotation (por seguridad)
    df = df.dropna(subset=["Annotation"])
    df = df[df["Annotation"].astype(str).str.strip() != ""]
    if df.empty:
        print(f"⚠️ {sitio}: no hay filas con Annotation válida.")
        return

    # Extraer lista de clusters únicos (ej. 'cluster_0', 'cluster_1', etc.)
    clusters = sorted(df["Annotation"].astype(str).unique())
    print(f"{sitio}: {len(clusters)} clusters encontrados en Annotation: {clusters}")

    # Carpeta del sitio
    carpeta_sitio = RUTA_SALIDA / sitio
    carpeta_sitio.mkdir(exist_ok=True)

    for cl in clusters:
        df_c = df[df["Annotation"].astype(str) == cl].copy()
        if df_c.empty:
            continue

        # Resetear Selection para que vaya de 1..N
        if "Selection" in df_c.columns:
            df_c = df_c.reset_index(drop=True)
            df_c["Selection"] = df_c.index + 1

        # Asegurar que Begin/End Time sean numéricos y válidos
        for col in ["Begin Time (s)", "End Time (s)"]:
            if col in df_c.columns:
                df_c[col] = pd.to_numeric(df_c[col], errors="coerce")

        if {"Begin Time (s)", "End Time (s)"}.issubset(df_c.columns):
            df_c = df_c.dropna(subset=["Begin Time (s)", "End Time (s)"])
            if df_c.empty:
                print(f"⚠️ {sitio} | {cl}: todas las filas tienen tiempos inválidos.")
                continue

        # ============================================================
        # 🔵 ***Opción 2: combinar cluster + archivo_origen en Annotation***
        # ============================================================
        if "archivo_origen" in df_c.columns:
            df_c["Annotation"] = (
                df_c["Annotation"].astype(str)
                + " | "
                + df_c["archivo_origen"].astype(str)
            )
        # ============================================================

        # --- Reordenar columnas para mantener archivo_origen y sitio visibles ---
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

        # Mantener también cualquier otra columna que venga de Fase 2
        otras = [c for c in df_c.columns if c not in columnas_base + extras]

        columnas_finales = [c for c in columnas_base if c in df_c.columns] + extras + otras
        df_c = df_c[columnas_finales]

        # Crear carpeta del cluster
        carpeta_cluster = carpeta_sitio / cl
        carpeta_cluster.mkdir(exist_ok=True)

        # Guardar archivo TSV compatible con Raven Lite
        out_path = carpeta_cluster / f"{sitio}_{cl}_selection_table.txt"
        df_c.to_csv(out_path, sep="\t", index=False)

        print(f"   ✓ {sitio} | {cl}: {len(df_c)} selecciones → {out_path}")

    print(f"✔️ Listo: Selection Tables por cluster generadas para {sitio}")


# ================================================================
# EJECUCIÓN MULTI-SITIO
# ================================================================

if __name__ == "__main__":
    for sitio, ruta_st in SELECTION_TABLES.items():
        split_selection_table_por_cluster(sitio, ruta_st)

    print("\n🎉 Proceso completado. Las Selection Tables por cluster están listas para Raven Lite.")