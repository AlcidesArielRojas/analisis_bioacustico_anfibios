# Resumen: Campaña Diciembre 2024 — Paisajes Sonoros Tapytá
Generado automáticamente el 2026-05-14

---

## ¿Dónde están los archivos pesados?

Los archivos de features originales (parquets ~99MB por sitio) y carpetas temporales (~112MB por sitio)
permanecen en:
  C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\resultados_HDD_Seagate\Campaña diciembre 2024\

Esta carpeta contiene COPIAS de: scripts, figuras, métricas y tablas clave (sin duplicar los 9.1 GB pesados).

---

## Estructura de esta carpeta

  01_Scripts/                     — 28 scripts Python usados en toda la campaña
  02_Resultados_por_sitio/        — Métricas, tablas UMAP, selection tables, figuras por sitio (20 sitios)
  03_Figuras_consolidadas/
      UMAP_2D/                    — 60 figuras PNG (scatter 2D HDBSCAN, todos los sitios)
      UMAP_3D_interactivo/        — 20 HTMLs interactivos 3D (abrir en browser)
      Overlay_BD/                 — 40 archivos PNG+HTML con proyección de la BD de anfibios
      PCA_varianza/               — 20 gráficos curva de varianza explicada por PCA
      Ruido_vs_valido/            — 20 gráficos proporción ruido HDBSCAN
      Subclustering/              — 178 figuras de subclustering (clusters mezclados)
  04_Metricas_globales/           — CSV resumen con métricas de los 265 clusters

---

## RESUMEN EJECUTIVO DEL ANÁLISIS

### Datos procesados
- Sitios: 20 (BO-31 a BO-40, Bo-38, EU-16 a EU-20, PA-16 a PA-20)
- Tipos de hábitat: Bosque (10 sitios), Eucaliptal (5), Pastizal (5)
- Segmentos de audio analizados: 1,852,242 (ventanas de 4s, solapamiento 2s)
- Filtro horario aplicado: 18:00 a 06:00 h
- Atenuación adaptativa: 6–12 kHz (reducción ruido insectos)

### Parámetros del pipeline
- MFCC: 20 coeficientes → 40 features (media + SD) por ventana
- PCA: umbral 95% varianza (promedio 24 componentes por sitio, rango 16–29)
- UMAP: n_neighbors=60, min_dist=0.3, 3 dimensiones, métrica coseno
- HDBSCAN: min_cluster_size=800, min_samples=65, método EOM, eps=0.03

---

## RESULTADOS POR FASE

### FASE 1: Extracción de MFCCs ✅ COMPLETADA — 20/20 sitios
- ~92,600 segmentos promedio por sitio (rango: 73,690 – 100,000)

### FASE 2: UMAP + HDBSCAN Clustering ✅ COMPLETADA — 20/20 sitios
- CLUSTERS GENERADOS: 265 en total (promedio 13.2 por sitio, rango 2–40)
- SILHOUETTE promedio: 0.394 (rango 0.063 – 0.591)
- DAVIES-BOULDIN promedio: 0.645
- RUIDO (segmentos no asignados) promedio: 6.8%

  Mejores sitios (Silhouette > 0.48):   EU-16Tapyta (0.591), Bo-38Tapyta (0.544),
                                         PA-16Tapyta (0.500), EU-17Tapyta (0.495)
  Sitios más difíciles (Silhouette < 0.25): BO-34 Tapyta (0.063), EU-19Tapyta (0.168)

  Por hábitat:
    Bosque:     8.1 clusters promedio, Silhouette 0.368, ruido 3.1%
    Eucaliptal: 3.6 clusters promedio, Silhouette 0.404, ruido 0.0%
    Pastizal:  33.2 clusters promedio, Silhouette 0.437, ruido 20.9%

  → Los pastizales tienen MÁS diversidad acústica (más clusters) pero también más
    ruido ambiente. Los eucaliptales son los más "limpios" acústicamente.

### FASE 3: Inspección visual y auditiva ✅ COMPLETADA — 20/20 sitios
- Figuras UMAP 2D + 3D interactivo generadas para todos los sitios
- Curvas PCA + gráficos ruido vs. válido disponibles

### FASE 4.1: Detección de clusters mezclados (5C) ✅ COMPLETADA — 20/20 sitios
- 277 clusters evaluados con detección avanzada de mezcla
- 71 clusters identificados como mezclados (25.6%)

### FASE 4.2: Subclustering jerárquico ✅ COMPLETADA — 20/20 sitios
- 161 de 265 clusters sometidos a subclustering (66.8%)
- Figuras de subclustering disponibles en 03_Figuras_consolidadas/Subclustering/

### OVERLAY BD: Proyección de base de datos en UMAP ✅ COMPLETADA — 20/20 sitios
- Figuras 2D + HTML 3D con posición de audios de referencia superpuesta en cada UMAP

### MATCHING YAMNet: Identificación automática de especies ✅ COMPLETADA — 20/20 sitios
- Representativos extraídos de clusters y comparados con BD de 53 especies (142 WAVs)
- 570 representativos con identificación top-1
- Similitud coseno media: 0.855 (mediana: 0.876) — ALTA CONFIANZA

  TOP 10 ESPECIES DETECTADAS (frecuencia de aparición en clusters):
    1. Leptodactylus elenae       — 163 detecciones (28.6%)
    2. Leptodactylus gracilis     —  75 detecciones (13.2%)
    3. Rhinella dorbignyi         —  52 detecciones ( 9.1%)
    4. Dendrosophus sanborni      —  51 detecciones ( 8.9%)
    5. Scinax fuscomarginatus     —  42 detecciones ( 7.4%)
    6. Hypsiboas curupi           —  38 detecciones ( 6.7%)
    7. Pithecopus sp.             —  19 detecciones ( 3.3%)
    8. Elachistocleis ovalis      —  18 detecciones ( 3.2%)
    9. Leptodactylus latrans      —  15 detecciones ( 2.6%)
   10. Boana faber                —  14 detecciones ( 2.5%)

  Similitud por hábitat:
    Pastizal:   sim media 0.868 — mejor matching
    Bosque:     sim media 0.852
    Eucaliptal: sim media 0.795 — más difícil identificar (menos diversidad?)

---

## ¿QUÉ FALTA PARA EL CONGRESO?

HECHO (pipeline técnico completo):
  [x] MFCC + UMAP + HDBSCAN en los 20 sitios
  [x] Subclustering de clusters mezclados
  [x] Matching YAMNet con base de datos de anfibios
  [x] Figuras por sitio y consolidadas

PENDIENTE para presentación científica:
  [ ] Umbral de confianza: filtrar matches con similitud < 0.85
  [ ] Tabla consolidada: sitio × especie × n_detecciones × sim_media
  [ ] Figura de diversidad por hábitat (riqueza de especies por tipo de ambiente)
  [ ] Abstract para SOLABIMA 2026 (texto plano + LaTeX, sin tablas ni figuras)

---

## SCRIPTS EN ESTA CARPETA (01_Scripts/)

PIPELINE PRINCIPAL:
  Fase1_Extracción_de_MFCCs...py    — Extracción de MFCCs con filtro horario y atenuación
  Fase1_5_Script_para_fusionar...py — Fusionar MFCCs de múltiples grabadoras del mismo sitio
  Fase2_UMAP_HDBSCAN.py             — Reducción dimensional y clustering
  Fase2_5_Generar_selection_table...py — Selection tables por cluster para Raven
  Fase3_Inspeccion_visual...py      — Exploración visual/auditiva interactiva

SUBCLUSTERING Y REFINAMIENTO:
  Fase_4_1_Deteccion_automatica...py    — Detectar clusters candidatos para subclustering
  Fase_4_2_Subclustering_jerarquico.py  — Subclustering con corte automático
  Fase_4_3_Visualizacion_subclusters.py — Figuras de subclustering
  Fase_4_X_Contexto_clusters...py       — Contexto de clusters en UMAP original
  Script_5B_Subclustering_automático.py — Subclustering estilo 5B
  Script_5B_hier_...py                  — Subclustering jerárquico

MATCHING CON BASE DE DATOS:
  Script_1_Conversion...WAV_uniforme.py     — Preparar BD de anfibios en WAV uniforme
  Script_2_Extraccion_embeddings_ref.py     — Embeddings YAMNet de la BD
  Script_3_Extraccion_embeddings_rep.py     — Embeddings YAMNet de representativos
  Script_4_Matching_automatico.py           — Matching coseno top-K
  Script_5_Espectrogramas_emparejados.py    — Visualizar pares representativo vs BD

PROYECCIÓN BD EN CLUSTERS:
  Fase_1_Extraccion_MFCC_BD.py            — MFCCs de la BD para proyectar en UMAP
  Script_1_Verificacion_modelos.py         — Verificar modelos Fase 2 guardados
  Script_2_Proyeccion_BD_en_UMAP.py       — Proyectar BD en el espacio UMAP ya entrenado
  Script_3_UMAP_overlay_2D_3D.py          — Figuras overlay BD
  Script_4_Asignacion_BD_a_clusters.py    — Asignar grabaciones BD a clusters HDBSCAN
  Script_5C_Deteccion_avanzada.py         — Métricas 5C de mezcla

VISUALIZACIÓN Y EXTRAS:
  UMAP_CANTERA_con_etiquetas_de_especies.py   — UMAP con etiquetas manuales de especie
  Fase4_Anotacion_automatica_selection.py     — Anotar selection tables con especie
  Comparacion_BD_con_Anfibios_de_Tapyta.py   — Comparación BD vs sitios Tapytá
  Script_Preparar_Tabla_Metrics_para_Latex.py — Tabla métricas formato LaTeX

---
Fin del resumen. Para regenerar cualquier resultado, usar los scripts de 01_Scripts/
con el entorno conda: paisajes_matching
