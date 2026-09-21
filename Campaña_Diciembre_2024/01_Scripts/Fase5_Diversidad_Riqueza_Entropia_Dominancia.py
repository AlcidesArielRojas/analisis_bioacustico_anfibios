# ================================================================
# Verificacion + calculo de Riqueza (Rs), Entropia (Hs), Dominancia (Ds)
# por sitio, y prueba de permutacion (10,000 replicas) comparando habitats.
#
# PASO 1 (verificacion): reconstruir el "tipo final" de cada segmento como
# (cluster_hdbscan, subcluster_id) cuando subcluster_id >= 0, o solo
# cluster_hdbscan cuando subcluster_id == -1. Contar tipos distintos
# (excluyendo ruido HDBSCAN, cluster_hdbscan == -1) y confirmar que da 265.
# Si NO da 265, el script se detiene y no calcula nada mas (para no
# reportar numeros que no cuadren con lo ya publicado).
# ================================================================
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import f_oneway

BASE = Path(r"C:/Users/User/Dropbox/Proyecto_Paisajes_Sonoros_Repositorio_Local/resultados_HDD_Seagate/Campaña diciembre 2024")
SUFIJO = "v2_horario18a06_insectos6a12"

SITIOS = [
    "BO-31Tapyta", "BO-32Tapyta", "BO-33Tapyta", "BO-34 Tapyta", "BO-35Tapyta",
    "BO-36Tapyta", "BO-37Tapyta", "Bo-38Tapyta", "BO-39Tapyta", "BO-40Tapyta",
    "EU-16Tapyta", "EU-17Tapyta", "EU-18Tapyta", "EU-19Tapyta", "EU-20Tapyta",
    "PA-16Tapyta", "PA-17Tapyta", "PA-18Tapyta", "PA-19Tapyta", "PA-20Tapyta",
]

HABITAT = {"BO": "Bosque", "EU": "Eucaliptal", "PA": "Pastizal"}

def habitat_de(sitio: str) -> str:
    return HABITAT[sitio[:2].upper()]

registros = []
total_tipos_finales = set()
faltantes = []

for sitio in SITIOS:
    ruta = BASE / sitio / f"{sitio}_{SUFIJO}_fase4_umap_hdbscan_subclusters.csv"
    if not ruta.exists():
        faltantes.append(sitio)
        continue
    df = pd.read_csv(ruta, usecols=["cluster_hdbscan", "subcluster_id"])

    # Excluir ruido HDBSCAN real (-1 en cluster_hdbscan)
    df = df[df["cluster_hdbscan"] != -1].copy()

    # Tipo final: (cluster_hdbscan, subcluster_id) si subcluster_id>=0, si no, solo cluster_hdbscan
    def tipo_final(row):
        if row["subcluster_id"] >= 0:
            return (int(row["cluster_hdbscan"]), int(row["subcluster_id"]))
        return (int(row["cluster_hdbscan"]), -1)

    df["tipo_final"] = list(zip(df["cluster_hdbscan"], df["subcluster_id"].where(df["subcluster_id"] >= 0, -1)))

    conteo = df["tipo_final"].value_counts()
    n_s = conteo.sum()
    q = conteo / n_s

    R_s = len(conteo)
    H_s = -np.sum(q * np.log(q))
    D_s = np.sum(q ** 2)  # Simpson dominance

    tipos_globales = {(sitio, t) for t in conteo.index}
    total_tipos_finales |= tipos_globales

    registros.append({
        "sitio": sitio,
        "habitat": habitat_de(sitio),
        "n_segmentos": int(n_s),
        "R_s": R_s,
        "H_s": H_s,
        "D_s": D_s,
    })

if faltantes:
    print("ADVERTENCIA - sitios sin archivo fase4_subclusters:", faltantes)

df_res = pd.DataFrame(registros)
print(df_res.to_string(index=False))

total_R = df_res["R_s"].sum()
print(f"\nSuma de R_s (tipos finales) en todos los sitios: {total_R}")
print("(comparar con el total publicado: 265)")

if total_R != 265:
    print("\n*** NO COINCIDE con 265. Me detengo aca, no calculo permutacion. ***")
    sys.exit(1)

print("\n*** COINCIDE con 265. Prosigo con la prueba de permutacion. ***\n")

# ---------------- Prueba de permutacion (10,000 replicas) ----------------
rng = np.random.default_rng(42)
N_PERM = 10000

def f_stat(valores, grupos):
    grupos_unicos = sorted(set(grupos))
    muestras = [np.array(valores)[np.array(grupos) == g] for g in grupos_unicos]
    stat, _ = f_oneway(*muestras)
    return stat

resultados_perm = {}
for metrica in ["R_s", "H_s", "D_s"]:
    valores = df_res[metrica].values
    grupos = df_res["habitat"].values
    obs = f_stat(valores, grupos)

    conteo_mayor_igual = 0
    for _ in range(N_PERM):
        grupos_mezclados = rng.permutation(grupos)
        stat = f_stat(valores, grupos_mezclados)
        if stat >= obs:
            conteo_mayor_igual += 1
    p_valor = conteo_mayor_igual / N_PERM
    resultados_perm[metrica] = (obs, p_valor)
    print(f"{metrica}: F observado = {obs:.4f}, p (permutacion, {N_PERM} reps) = {p_valor:.4f}")

print("\n--- Promedios por habitat ---")
print(df_res.groupby("habitat")[["R_s", "H_s", "D_s"]].mean().round(4))
