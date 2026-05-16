# ================================================================
# Fusión de features Fase 1: BO-81Tapyta + BO-82Tapyta → BO-Tapyta
# Sin recalcular MFCCs, solo combinando parquet
# ================================================================

from pathlib import Path
import pandas as pd

BASE_DIR = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local")
ruta_salida = BASE_DIR / "resultados"

SUFIJO_CORRIDA = "v2_horario18a06_insectos6a12"

SITE_A = "BO-81Tapyta"
SITE_B = "BO-82Tapyta"
SITE_FUSION = "BO-Tapyta"  # nombre nuevo para el sitio combinado

ruta_a = ruta_salida / f"features_{SITE_A}_{SUFIJO_CORRIDA}.parquet"
ruta_b = ruta_salida / f"features_{SITE_B}_{SUFIJO_CORRIDA}.parquet"

ruta_fusion = ruta_salida / f"features_{SITE_FUSION}_{SUFIJO_CORRIDA}.parquet"

print("Leyendo:")
print(" -", ruta_a)
print(" -", ruta_b)

df_a = pd.read_parquet(ruta_a)
df_b = pd.read_parquet(ruta_b)

print(f"Segmentos {SITE_A}: {len(df_a)}")
print(f"Segmentos {SITE_B}: {len(df_b)}")

# Homogeneizar nombre de sitio
if "sitio" in df_a.columns:
    df_a["sitio"] = SITE_FUSION
if "sitio" in df_b.columns:
    df_b["sitio"] = SITE_FUSION

# Concatenar
df_fusion = pd.concat([df_a, df_b], ignore_index=True)
print(f"Total segmentos fusionados ({SITE_FUSION}): {len(df_fusion)}")

# Guardar
df_fusion.to_parquet(ruta_fusion, index=False)
print(f"💾 Guardado fusionado en: {ruta_fusion}")