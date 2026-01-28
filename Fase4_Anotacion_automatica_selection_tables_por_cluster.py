# ================================================================
# Script: Anotación automática de Selection Tables por cluster
# Autor: Alcides Ariel Rojas Geraldo
# Contacto: alcidesrojasg@gmail.com
# Fecha: 27/01/2026
#
# Descripción:
# Este script toma las Selection Tables por cluster generadas en Fase 2
# para un sitio específico (p. ej., CANTERA1Tapy) y asigna de forma
# automática una etiqueta ecológica (nombre de especie) a todos los
# segmentos pertenecientes a los clusters previamente identificados.
#
# La anotación se realiza directamente sobre las Selection Tables
# (segmentos), sin modificar los audios originales. El script:
#   - Lee cada carpeta de cluster (cluster_0, cluster_1, ...)
#   - Verifica si ese cluster tiene una especie asignada en el diccionario
#   - Reemplaza la columna "Annotation" por el nombre de la especie
#   - Conserva el cluster original en una columna adicional
#   - Guarda las Selection Tables anotadas en una carpeta separada
#
# Este proceso permite avanzar hacia la fase final de anotación ecológica
# de forma reproducible, ordenada y completamente automatizada.
# ================================================================

import pandas as pd
from pathlib import Path

# =========================
# CONFIGURACIÓN
# =========================

BASE_DIR = Path(r"C:/Users/User/Proyecto_Paisajes_Sonoros_Repositorio_Local")

SITIO = "CANTERA1Tapy"

RUTA_ST_ORIG = BASE_DIR / "selection_tables_por_cluster_desde_raven" / SITIO
RUTA_ST_ANOT = BASE_DIR / "selection_tables_por_cluster_anotadas" / SITIO
RUTA_ST_ANOT.mkdir(parents=True, exist_ok=True)

# Diccionario cluster -> especie (ejemplos, completás vos)
CLUSTER_A_ESPECIE = {
    "cluster_28": "Dendropsophus minutus",
    "cluster_27": "Dendropsophus minutus",
    "cluster_26": "Dendropsophus spp.",
    "cluster_24": "Dendropsophus minutus",
    "cluster_19": "Dendropsophus cf. minutus",
    "cluster_18": "Dendropsophus minutus",
    "cluster_12": "Dendropsophus cf. minutus",
    "cluster_9": "Dendropsophus cf. minutus",
    "cluster_3": "Dendropsophus spp.",
}

# Si querés que en Annotation quede "Boana caingua | cluster_3"
INCLUIR_CLUSTER_EN_ANNOTATION = True

# =========================
# FUNCIÓN PRINCIPAL
# =========================

def anotar_selection_tables_sitio(sitio: str):
    carpeta_sitio = RUTA_ST_ORIG
    if not carpeta_sitio.exists():
        print(f"⚠️ No existe carpeta de ST por cluster para {sitio}: {carpeta_sitio}")
        return

    clusters = sorted(
        [p.name for p in carpeta_sitio.iterdir() if p.is_dir() and p.name.startswith("cluster_")]
    )

    print(f"\nSitio: {sitio}")
    print(f"Clusters encontrados: {clusters}")

    for cl in clusters:
        ruta_cluster = carpeta_sitio / cl
        nombre_st = f"{sitio}_{cl}_selection_table.txt"
        ruta_st = ruta_cluster / nombre_st

        if not ruta_st.exists():
            print(f"   ⚠️ No se encontró ST para {sitio} | {cl}: {ruta_st}")
            continue

        if cl not in CLUSTER_A_ESPECIE:
            print(f"   · {cl}: sin especie asignada (se deja igual).")
            continue

        especie = CLUSTER_A_ESPECIE[cl]
        print(f"   ✓ {cl}: anotando como '{especie}'")

        df = pd.read_csv(ruta_st, sep="\t")

        if "cluster_original" not in df.columns:
            df["cluster_original"] = cl

        if INCLUIR_CLUSTER_EN_ANNOTATION:
            df["Annotation"] = especie + " | " + cl
        else:
            df["Annotation"] = especie

        carpeta_cluster_out = RUTA_ST_ANOT / cl
        carpeta_cluster_out.mkdir(parents=True, exist_ok=True)

        ruta_out = carpeta_cluster_out / nombre_st
        df.to_csv(ruta_out, sep="\t", index=False)
        print(f"      → Guardado: {ruta_out}")

    print("\n✔️ Anotación automática completada para", sitio)


if __name__ == "__main__":
    anotar_selection_tables_sitio(SITIO)