# ================================================================
# Script 3 NUEVO: UMAP overlay 2D y 3D con audios BD (coordenadas reales)
# ------------------------------------------------
# - Carga Fase 2 (campaña) por sitio
# - Carga proyección BD UMAP real por sitio
# - Genera figuras 2D y 3D con overlay
# ================================================================

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

NOMBRE_CAMPANIA = "Campaña diciembre 2024"
SUFIJO_CORRIDA = "v2_horario18a06_insectos6a12"

BASE_DIR = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local")
BASE_RESULTADOS = BASE_DIR / "resultados_HDD_Seagate" / NOMBRE_CAMPANIA
RUTA_PROYECCION_BD = BASE_DIR / "proyeccion_BD_fase1_5"

sns.set_theme(context="talk", style="whitegrid", palette="tab20")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["savefig.dpi"] = 200


def overlay_sitio(sitio: str):
    print(f"\n=== Overlay UMAP campaña + BD para sitio: {sitio} ===")

    ruta_sitio = BASE_RESULTADOS / sitio
    ruta_fase2 = ruta_sitio / f"{sitio}_{SUFIJO_CORRIDA}_fase2_umap_hdbscan.csv"
    ruta_bd = RUTA_PROYECCION_BD / f"{sitio}_{SUFIJO_CORRIDA}_proyeccion_BD_umap_real.csv"

    if not ruta_fase2.exists():
        print(f"⚠️ No se encontró Fase 2 para {sitio}: {ruta_fase2}")
        return
    if not ruta_bd.exists():
        print(f"⚠️ No se encontró proyección BD para {sitio}: {ruta_bd}")
        return

    df_f2 = pd.read_csv(ruta_fase2)
    df_bd = pd.read_csv(ruta_bd)

    # 2D overlay: campaña (fondo) + BD (puntos destacados)
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(top=0.9, bottom=0.1)

    # Campaña
    sns.scatterplot(
        data=df_f2, x="DIM1", y="DIM2",
        hue="cluster_hdbscan",
        palette="tab20", s=10, linewidth=0, alpha=0.3, ax=ax,
        legend=False
    )

    # BD (coordenadas reales)
    ax.scatter(
        df_bd["BD_DIM1"], df_bd["BD_DIM2"],
        c="red", s=30, alpha=0.9, label="BD (UMAP real)", edgecolors="k", linewidths=0.3
    )

    ax.set_title(f"{sitio} | UMAP 2D campaña + BD (coordenadas reales)")
    ax.set_xlabel("DIM1")
    ax.set_ylabel("DIM2")
    ax.legend(loc="upper right")
    plt.tight_layout()

    figuras_dir = ruta_sitio / "figuras_overlay_bd"
    figuras_dir.mkdir(parents=True, exist_ok=True)
    ruta_fig = figuras_dir / f"{sitio}_{SUFIJO_CORRIDA}_umap_overlay_bd_2d.png"
    fig.savefig(ruta_fig)
    plt.close(fig)
    print(f"💾 Figura 2D guardada en: {ruta_fig}")

    # 3D overlay con Plotly
    fig3d = px.scatter_3d(
        df_f2, x="U1", y="U2", z="U3",
        color="cluster_hdbscan",
        opacity=0.3,
        title=f"{sitio} | UMAP 3D campaña + BD (coordenadas reales)"
    )

    fig3d_bd = px.scatter_3d(
        df_bd, x="BD_U1", y="BD_U2", z="BD_U3",
        color_discrete_sequence=["red"],
        opacity=0.9
    )

    # Agregar trazas de BD al fig3d
    for trace in fig3d_bd.data:
        trace.name = "BD (UMAP real)"
        fig3d.add_trace(trace)

    ruta_html = figuras_dir / f"{sitio}_{SUFIJO_CORRIDA}_umap_overlay_bd_3d.html"
    fig3d.write_html(str(ruta_html))
    print(f"💾 Figura 3D interactiva guardada en: {ruta_html}")


def main():
    grabadoras = [p.name for p in BASE_RESULTADOS.iterdir() if p.is_dir()]
    grabadoras = sorted(grabadoras)

    print("Sitios detectados para overlay BD:")
    for g in grabadoras:
        print(" -", g)

    for sitio in grabadoras:
        overlay_sitio(sitio)

    print("\n✅ Overlay BD completado.")


if __name__ == "__main__":
    main()
