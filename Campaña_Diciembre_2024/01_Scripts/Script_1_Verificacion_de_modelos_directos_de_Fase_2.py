# ================================================================
# Script 1 NUEVO: Verificación de modelos directos de Fase 2
# ------------------------------------------------
# - Carga modelos_directos_{SITE}_{SUFIJO_CORRIDA}.pkl
# - Muestra un resumen por sitio (componentes PCA, params UMAP)
# - NO reentrena nada, solo verifica que todo esté en su lugar.
# ================================================================

from pathlib import Path
import joblib

NOMBRE_CAMPANIA = "Campaña diciembre 2024"
SUFIJO_CORRIDA = "v2_horario18a06_insectos6a12"

BASE_DIR = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local")
RUTA_MODELOS = BASE_DIR / "modelos_fase2_fase4" / NOMBRE_CAMPANIA

def verificar_modelo_sitio(sitio: str):
    ruta_modelo = RUTA_MODELOS / f"modelos_directos_{sitio}_{SUFIJO_CORRIDA}.pkl"
    if not ruta_modelo.exists():
        print(f"⚠️ No se encontró modelo directo para {sitio}: {ruta_modelo}")
        return

    modelo = joblib.load(ruta_modelo)
    print(f"\n=== Modelo para sitio: {sitio} ===")
    print(f"  Campaña: {modelo.get('campania')}")
    print(f"  Sufijo corrida: {modelo.get('sufijo_corrida')}")
    print(f"  # features usados: {len(modelo.get('features', []))}")
    print(f"  PCA componentes: {modelo.get('pca_n_components')}")
    print(f"  PCA varianza explicada: {modelo.get('pca_variance_explained'):.3f}")
    umap_params = modelo.get("umap_params", {})
    print(f"  UMAP n_neighbors: {umap_params.get('n_neighbors')}")
    print(f"  UMAP min_dist: {umap_params.get('min_dist')}")
    print(f"  UMAP metric: {umap_params.get('metric')}")
    print(f"  UMAP densmap: {umap_params.get('densmap')}")


def main():
    if not RUTA_MODELOS.exists():
        print(f"⚠️ Carpeta de modelos no existe: {RUTA_MODELOS}")
        return

    sitios = sorted([
        p.name for p in RUTA_MODELOS.iterdir()
        if p.is_file() and p.name.startswith("modelos_directos_") and p.name.endswith(f"{SUFIJO_CORRIDA}.pkl")
    ])

    if not sitios:
        print("⚠️ No se encontraron modelos directos. ¿Corriste Fase 2 corregida?")
        return

    print("Modelos directos encontrados:")
    for fname in sitios:
        print(" -", fname)

    # Extraer nombres de sitio
    sitios_unicos = sorted({fname.split("modelos_directos_")[1].split(f"_{SUFIJO_CORRIDA}.pkl")[0]
                            for fname in sitios})

    for sitio in sitios_unicos:
        verificar_modelo_sitio(sitio)

    print("\n✅ Verificación de modelos completada.")


if __name__ == "__main__":
    main()
