# ================================================================
# Script: Comparación de especies entre la Base de Datos de Anfibios
#         y el listado del libro "Anfibios de la Reserva Natural Tapytá"
# ------------------------------------------------
# Este script:
#   1) Lee automáticamente todas las especies presentes en la base de datos
#      local de anfibios (carpetas + audios de "Cantos Paraguay").
#   2) Normaliza los nombres científicos para evitar diferencias por mayúsculas,
#      espacios o formatos.
#   3) Compara esa lista contra el listado oficial del libro de Tapytá.
#   4) Identifica:
#        - qué especies del libro NO están en tu base de datos,
#        - qué especies de tu base de datos NO aparecen en el libro.
#   5) Genera un archivo CSV con el resumen completo de coincidencias y ausencias.
#
# Este proceso permite verificar la cobertura de tu base de referencia antes
# de avanzar con la identificación automática de cantos.
# ================================================================

from pathlib import Path
import pandas as pd

# -------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------

BASE_DIR = Path(r"C:/Users/User/Proyecto_Paisajes_Sonoros_Repositorio_Local")
BD_ANFIBIOS = BASE_DIR / "BD anfibios"
CANTOS_PY = BD_ANFIBIOS / "Cantos Paraguay"

# Lista de especies del libro "Anfibios de la Reserva Natural Tapytá"
ESPECIES_TAPYTA = [
    # Hylidae
    "Scinax acuminatus",
    "Ololygon berthae",
    "Scinax fuscovarius",
    "Scinax squalirostris",
    "Scinax nasicus",
    "Boana albopunctata",
    "Boana caingua",
    "Boana faber",
    "Boana raniceps",
    "Dendropsophus minutus",
    "Dendropsophus nanus",
    "Dendropsophus sanborni",
    "Itapotihyla langsdorffii",
    "Trachycephalus typhonius",
    # Phyllomedusidae
    "Pithecopus azureus",
    # Leptodactylidae
    "Adenomera diptyx",
    "Physalaemus albonotatus",
    "Physalaemus cuvieri",
    "Leptodactylus elenae",
    "Leptodactylus fuscus",
    "Leptodactylus gracilis",
    "Leptodactylus latrans",
    "Leptodactylus mystacinus",
    "Leptodactylus podicipinus",
    # Microhylidae
    "Elachistocleis bicolor",
    # Odontophrynidae
    "Odontophrynus americanus",
    "Proceratophrys avelinoi",
    # Bufonidae
    "Melanophryniscus devincenzii",
    "Rhinella diptycha",
    "Rhinella ornata",
    "Rhinella azarai",
]

# -------------------------------------------------
# NORMALIZACIÓN DE NOMBRES
# -------------------------------------------------

def norm_nombre(s: str) -> str:
    """Normaliza nombres de especies para comparación."""
    return " ".join(s.strip().lower().split())


# -------------------------------------------------
# LECTURA DE ESPECIES EN BD ANFIBIOS
# -------------------------------------------------

def listar_especies_por_carpeta():
    especies = []
    for p in BD_ANFIBIOS.iterdir():
        if p.is_dir() and p.name != "Cantos Paraguay":
            especies.append(p.name.strip())
    return sorted(especies)


def listar_especies_cantos_paraguay():
    especies = set()
    if not CANTOS_PY.exists():
        print(f"⚠️ No existe carpeta Cantos Paraguay: {CANTOS_PY}")
        return []

    for f in CANTOS_PY.iterdir():
        if not f.is_file():
            continue
        # Ejemplo: "08-A08 - Dendropsophus nanus.wma"
        nombre = f.stem  # sin extensión
        if " - " in nombre:
            _, especie = nombre.split(" - ", 1)
            especies.add(especie.strip())
        else:
            # Si algún archivo no sigue el patrón, lo dejamos tal cual para revisar
            especies.add(nombre.strip())

    return sorted(especies)


# -------------------------------------------------
# COMPARACIÓN BD vs LIBRO
# -------------------------------------------------

def main():
    # Especies en BD (carpetas)
    especies_carpetas = listar_especies_por_carpeta()
    # Especies en Cantos Paraguay (a partir del nombre del archivo)
    especies_cantos_py = listar_especies_cantos_paraguay()

    # Unión de todo lo que tenés en la BD
    especies_bd_raw = set(especies_carpetas) | set(especies_cantos_py)

    # Normalizar
    bd_norm_map = {norm_nombre(e): e for e in especies_bd_raw}
    tapyta_norm_map = {norm_nombre(e): e for e in ESPECIES_TAPYTA}

    bd_norm = set(bd_norm_map.keys())
    tapyta_norm = set(tapyta_norm_map.keys())

    # Especies del libro que NO están en la BD
    faltan_en_bd_norm = tapyta_norm - bd_norm
    # Especies de la BD que NO están en el libro
    sobran_en_bd_norm = bd_norm - tapyta_norm

    print("\n=== Especies en BD (carpetas) ===")
    for e in sorted(especies_carpetas):
        print(" -", e)

    print("\n=== Especies en BD (Cantos Paraguay) ===")
    for e in sorted(especies_cantos_py):
        print(" -", e)

    print("\n=== Especies del libro Tapytá ===")
    for e in ESPECIES_TAPYTA:
        print(" -", e)

    print("\n=== Especies del libro que NO están en la BD ===")
    for k in sorted(faltan_en_bd_norm):
        print(" -", tapyta_norm_map[k])

    print("\n=== Especies en la BD que NO están en el libro ===")
    for k in sorted(sobran_en_bd_norm):
        print(" -", bd_norm_map[k])

    # Guardar resumen en CSV
    filas = []

    for e in especies_bd_raw:
        filas.append({
            "fuente": "BD_anfibios",
            "especie_raw": e,
            "especie_norm": norm_nombre(e),
            "esta_en_libro": norm_nombre(e) in tapyta_norm,
        })

    for e in ESPECIES_TAPYTA:
        filas.append({
            "fuente": "Libro_Tapyta",
            "especie_raw": e,
            "especie_norm": norm_nombre(e),
            "esta_en_bd": norm_nombre(e) in bd_norm,
        })

    df = pd.DataFrame(filas)
    ruta_out = BASE_DIR / "comparacion_especies_bd_vs_tapyta.csv"
    df.to_csv(ruta_out, index=False, encoding="utf-8")
    print(f"\n✓ Resumen detallado guardado en: {ruta_out}")


if __name__ == "__main__":
    main()
