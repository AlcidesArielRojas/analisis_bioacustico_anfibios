# ================================================================
# Script de inspección de Fase 1
# ------------------------------------------------
# Este código sirve para:
#   - mirar qué hay dentro de un lote .parquet de Fase 1,
#   - revisar todos los parquets temporales de una grabadora
#     (cuántos segmentos tienen y de qué archivos vienen),
#   - ver qué archivos WAV originales caen en un lote concreto,
#   - comprobar si las carpetas de la campaña y de una grabadora
#     existen correctamente en el disco externo.
#
# Es un script de diagnóstico: no modifica datos, solo imprime
# información útil para entender y depurar el procesamiento.
# ================================================================


import pandas as pd

# Ruta al archivo parquet
ruta = r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local\resultados_HDD_Seagate\Campaña diciembre 2024\BO-40Tapyta\temporales_fase1_BO-40Tapyta_v2_horario18a06_insectos6a12\lote_BO-40Tapyta_v2_horario18a06_insectos6a12_004.parquet"

# Leer parquet
df = pd.read_parquet(ruta)

# Mostrar primeras filas
print(df.head())

# Mostrar información general
print(df.info())

# Cantidad de filas
print("Total de segmentos:", len(df))

print(df.tail())

print(df['archivo_origen'].nunique())


from pathlib import Path
import pandas as pd

# Ruta a la carpeta de temporales de una grabadora
ruta_temporales = Path(
    r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local\resultados_HDD_Seagate\Campaña diciembre 2024\BO-40Tapyta\temporales_fase1_BO-40Tapyta_v2_horario18a06_insectos6a12"
)

parquets = sorted(ruta_temporales.glob("*.parquet"))

for p in parquets:
    df = pd.read_parquet(p)
    print(f"\nArchivo: {p.name}")
    print(f"  Segmentos: {len(df)}")
    print(f"  Archivos origen únicos: {df['archivo_origen'].nunique()}")
    print(f"  Primeros archivos origen:")
    print(df['archivo_origen'].head())



import numpy as np
from pathlib import Path

# Ruta a la carpeta Data
data_dir = Path(r"D:\Campaña diciembre 2024\BO-40Tapyta\Data")

# Lista de todos los WAV
archivos = sorted(data_dir.glob("*.wav"))

# Elegir lote N
N = 4
TAMANO_BLOQUE = 100

bloques = np.array_split(archivos, len(archivos) // TAMANO_BLOQUE)
lote = bloques[N-1]

for r in lote:
    print(r.name)


###################################################

from pathlib import Path

NOMBRE_CAMPANIA = "Campaña diciembre 2024"
DISCO_EXTERNO = Path(r"D:\\")

ruta_campania = DISCO_EXTERNO / NOMBRE_CAMPANIA
print("ruta_campania:", ruta_campania, "| exists:", ruta_campania.exists())

# Grabadora problemática
sitio = "Bo-38Tapyta"
carpeta = ruta_campania / sitio
data_dir = carpeta / "Data"

print("carpeta grabadora:", carpeta, "| exists:", carpeta.exists())
print("data_dir:", data_dir, "| exists:", data_dir.exists())

if carpeta.exists():
    print("\nContenido de la carpeta de la grabadora:")
    for p in carpeta.iterdir():
        print(" -", repr(p.name))
