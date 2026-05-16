import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local")
R = BASE_DIR / "resultados"

sitios = ["BO-Tapyta", "CANTERA1Tapy", "EU-41Tapyta", "PA-41Tapyta"]
sufijo = "v2_horario18a06_insectos6a12"

filas = []
for s in sitios:
    ruta = R / f"{s}_{sufijo}_fase2_umap_hdbscan_metrics.csv"
    df = pd.read_csv(ruta)
    filas.append({
        "Sitio": df.loc[0, "site"],
        "n_clusters": int(df.loc[0, "n_clusters"]),
        "proporcion_ruido": df.loc[0, "proporcion_ruido"],
        "silhouette": df.loc[0, "silhouette"],
    })

tabla = pd.DataFrame(filas)
print(tabla)


print(tabla.to_latex(index=False, float_format="%.2f"))
