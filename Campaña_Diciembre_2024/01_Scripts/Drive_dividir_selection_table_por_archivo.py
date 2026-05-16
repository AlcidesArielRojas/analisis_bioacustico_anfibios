#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script: dividir_selection_table_por_archivo.py

Descripción general (paso a paso):
1. Lee un Selection Table de Raven (formato .txt, separado por tabuladores) que
   corresponde a un único cluster (por ejemplo, cluster_18).
2. A partir de la columna `archivo_origen`, identifica qué filas provienen del
   mismo archivo .wav.
3. Para cada archivo_origen:
   - Filtra sólo las filas correspondientes a ese archivo.
   - Elimina las columnas `archivo_origen`, `sitio` y `cluster_original`.
   - Reenumera la columna `Selection` desde 1 hasta N para ese archivo.
   - Reemplaza el contenido de la columna `Annotation`:
       * Si contiene "Dendropsophus minutus" (o "Dendropsophus minutus | cluster_18"),
         se reemplaza por "Den_minutus".
   - Construye un nombre de salida con el formato:
         SITIO_AÑOMESDIA_HHMMSS.Table.1.selections.txt
     donde:
       * SITIO se toma de la columna `sitio`.
       * AÑOMESDIA_HHMMSS se extrae del nombre del archivo .wav
         (por ejemplo: CANTERA1TAPY_20250603_180802.wav → 20250603_180802).
4. Guarda cada subconjunto en un archivo .txt separado (tabulado), listo para
   ser usado en Raven.

Uso:
- Ajustar la ruta de `INPUT_FILE` y `OUTPUT_DIR` según tu entorno.
"""

from pathlib import Path
import pandas as pd

# === CONFIGURACIÓN DE ENTRADA/SALIDA ===
INPUT_FILE = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local\selection_tables_por_cluster_anotadas\CANTERA1Tapy\cluster_24\CANTERA1Tapy_cluster_24_selection_table.txt")
OUTPUT_DIR = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local\selection_tables_por_cluster_anotadas\CANTERA1Tapy\salida_selection_tables_por_archivo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extraer_fecha_hora_de_wav(nombre_wav: str) -> str:
    """
    A partir de un nombre de archivo tipo:
        CANTERA1TAPY_20250603_180802.wav
    devuelve:
        20250603_180802
    """
    base = Path(nombre_wav).name  # CANTERA1TAPY_20250603_180802.wav
    sin_ext = base.replace(".wav", "")
    partes = sin_ext.split("_")
    if len(partes) >= 3:
        # asumiendo formato: SITIO_YYYYMMDD_HHMMSS
        fecha = partes[-2]
        hora = partes[-1]
        return f"{fecha}_{hora}"
    else:
        # fallback: devuelve todo sin extensión
        return sin_ext


def main():
    # 1. Leer el selection table de entrada
    df = pd.read_csv(INPUT_FILE, sep="\t")

    # Verificamos que existan las columnas necesarias
    columnas_necesarias = {"archivo_origen", "sitio", "Annotation"}
    faltantes = columnas_necesarias - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas necesarias en el archivo de entrada: {faltantes}")

    # 2. Iterar por cada archivo_origen
    for archivo, df_sub in df.groupby("archivo_origen"):
        df_sub = df_sub.copy()

        # 3a. Reemplazar Annotation por "Den_minutus" cuando corresponda
        df_sub["Annotation"] = df_sub["Annotation"].apply(
            lambda x: "Den_minutus" if "Dendropsophus minutus" in str(x) else x
        )

        # 3b. Reenumerar Selection desde 1
        if "Selection" in df_sub.columns:
            df_sub["Selection"] = range(1, len(df_sub) + 1)

        # 3c. Obtener sitio (asumimos que es constante dentro del grupo)
        sitio_valores = df_sub["sitio"].unique()
        if len(sitio_valores) != 1:
            raise ValueError(f"Se encontraron múltiples sitios en el mismo archivo_origen: {sitio_valores}")
        sitio = sitio_valores[0]

        # 3d. Extraer fecha y hora del nombre .wav dentro de archivo_origen
        #     Ejemplo: CANTERA1Tapy/CANTERA1TAPY_20250603_180802.wav
        wav_path = archivo  # puede venir con subcarpeta
        fecha_hora = extraer_fecha_hora_de_wav(wav_path)

        # 3e. Eliminar columnas que no queremos en la salida
        columnas_a_eliminar = [c for c in ["archivo_origen", "sitio", "cluster_original"] if c in df_sub.columns]
        df_sub = df_sub.drop(columns=columnas_a_eliminar)

        # 3f. Construir nombre de archivo de salida
        #     Formato: SITIO_AÑOMESDIA_HHMMSS.Table.1.selections.txt
        nombre_salida = f"{sitio}_{fecha_hora}.Table.1.selections.txt"
        ruta_salida = OUTPUT_DIR / nombre_salida

        # 3g. Guardar como .txt separado por tabuladores
        df_sub.to_csv(ruta_salida, sep="\t", index=False)
        print(f"Guardado: {ruta_salida}")

    print("Proceso completado.")


if __name__ == "__main__":
    main()