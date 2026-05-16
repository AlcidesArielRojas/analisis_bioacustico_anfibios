from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ==========================
# Configuración básica
# ==========================
BASE_DIR = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local")
R = BASE_DIR / "resultados"

SITE = "CANTERA1Tapy"
SUFIJO = "v2_horario18a06_insectos6a12"
ruta_umap = R / f"{SITE}_{SUFIJO}_fase2_umap_hdbscan.csv"

# ==========================
# Cargar datos
# ==========================
df = pd.read_csv(ruta_umap)
for col in ["DIM1", "DIM2", "cluster_hdbscan"]:
    if col not in df.columns:
        raise ValueError(f"Falta la columna {col} en {ruta_umap}")

# ==========================
# Diccionario cluster → especie
# ==========================
CLUSTER_A_ESPECIE = {
    28: "D. minutus",
    27: "D. minutus",
    26: "Dendropsophus spp.",
    24: "D. minutus",
    19: "D. cf. minutus",
    18: "D. minutus",
    12: "D. cf. minutus",
    9:  "D. cf. minutus",
    3:  "Dendropsophus spp."
}
ORDEN_ESPECIES = ["D. minutus", "D. cf. minutus", "Dendropsophus spp."]

# ==========================
# Preparar datos
# ==========================
df = df[df["cluster_hdbscan"] != -1]  # eliminar ruido

# Crear columna de color: anotados = color real, resto = gris
clusters_unicos = sorted(df["cluster_hdbscan"].unique())
palette = sns.color_palette("tab20", n_colors=len(clusters_unicos))
palette_dict = {c: palette[i] for i, c in enumerate(clusters_unicos)}

df["color"] = df["cluster_hdbscan"].apply(
    lambda c: palette_dict[c] if c in CLUSTER_A_ESPECIE else "lightgray"
)

# ==========================
# Figura UMAP + HDBSCAN
# ==========================
sns.set_theme(context="talk", style="whitegrid")
plt.rcParams["savefig.dpi"] = 300
fig, ax = plt.subplots(figsize=(10, 8))

# Scatterplot manual por color
ax.scatter(
    df["DIM1"], df["DIM2"],
    c=df["color"],
    s=22, alpha=0.9, linewidth=0
)

# Título y ejes
n_clusters = len(set(df["cluster_hdbscan"]))
ax.set_title(f"{SITE} | UMAP + HDBSCAN (sin ruido) | clusters={n_clusters}")
ax.set_xlabel("DIM1")
ax.set_ylabel("DIM2")
ax.set_aspect("auto")

# Ocultar leyenda automática
if ax.legend_ is not None:
    ax.legend_.remove()

# ==========================
# Etiquetas sobre clusters (solo número)
# ==========================
for cluster_id in CLUSTER_A_ESPECIE.keys():
    puntos = df[df["cluster_hdbscan"] == cluster_id]
    if len(puntos) == 0:
        continue
    x_med = puntos["DIM1"].median()
    y_med = puntos["DIM2"].median()
    ax.text(
        x_med, y_med, str(cluster_id),
        fontsize=12, weight="bold",
        color="black",
        ha="center", va="center",
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.3")
    )

# ==========================
# Leyenda abajo con puntos de color
# ==========================
items_ordenados = []
for especie in ORDEN_ESPECIES:
    for cluster_id, esp in CLUSTER_A_ESPECIE.items():
        if esp == especie:
            items_ordenados.append((especie, cluster_id))

handles = []
labels = []
for especie, cluster_id in items_ordenados:
    color = palette_dict.get(cluster_id, "gray")
    handles.append(
        Line2D([0], [0], marker="o", linestyle="", markersize=9,
               markerfacecolor=color, markeredgecolor="black")
    )
    labels.append(f"{especie} – cluster {cluster_id}")

ax.legend(
    handles,
    labels,
    title="Clusters anotados",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    frameon=False,
    ncol=3,
    fontsize=11,
    title_fontsize=11
)

plt.tight_layout(rect=(0, 0.08, 1, 1))

# ==========================
# Guardar figura
# ==========================
salida_png = R / f"{SITE}_{SUFIJO}_umap_hdbscan_anotado_sin_ruido.png"
plt.savefig(salida_png, bbox_inches="tight")
plt.close(fig)

print(f"✅ Figura anotada guardada en: {salida_png}")