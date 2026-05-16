# Proyecto Paisajes Sonoros — Bosque Atlántico Interior, Tapytá, Paraguay
> Este archivo es leído automáticamente por Claude Code al inicio de cada sesión.
> Contiene todo el contexto necesario para retomar el trabajo sin perder tiempo.

---

## 🎯 ¿De qué trata este proyecto?

**Monitoreo Acústico Pasivo (PAM)** de anfibios en el Bosque Atlántico Interior de Tapytá, Paraguay.

El pipeline analiza grabaciones de audio nocturnas registradas con grabadoras autónomas (AudioMoth), extrae características acústicas (MFCCs), aplica reducción dimensional (UMAP) y agrupamiento automático (HDBSCAN) para identificar y clasificar tipos de sonidos sin supervisión humana.

**Objetivo principal:** Presentar resultados en el congreso **SOLABIMA 2026** (Sociedad Latinoamericana de Biología Matemática).

**Contexto ecológico:** Las grabaciones corresponden al inicio de la temporada húmeda subtropical (noviembre–diciembre), que coincide con el pico de actividad vocal de los anfibios.

---

## 👤 Usuario y preferencias generales

- Comunicación: **español**, en tono claro y accesible (el usuario no es programador experto)
- Explicaciones técnicas: **siempre con analogías y lenguaje sencillo** cuando se introducen conceptos nuevos
- Idioma del código Python: **inglés** (variables, comentarios, nombres de funciones)
- Figuras académicas: **300 DPI, fondo blanco, sin emojis en títulos de figuras**

---

## 💻 Entornos Python — REGLAS CRÍTICAS

### Entorno principal (para TODO el análisis):
```
Nombre:     paisajes_matching
Ruta:       /c/Users/User/miniconda3/envs/paisajes_matching/python
```
**Cómo correr scripts:**
```bash
PYTHONUTF8=1 /c/Users/User/miniconda3/envs/paisajes_matching/python nombre_script.py
```
**Cómo correr código inline:**
```bash
PYTHONUTF8=1 /c/Users/User/miniconda3/envs/paisajes_matching/python -c "..."
```
El prefijo `PYTHONUTF8=1` es **obligatorio** en Windows para evitar errores de codificación con tildes y ñ en rutas.

### Entorno base (solo para PDF con pypdf):
```
Ruta:  /c/Users/User/miniconda3/python
```
Usar únicamente cuando se necesite `pypdf` o `pdfplumber`, que no están en `paisajes_matching`.

### ⚠️ Nunca usar:
- `python3` o `python` solos → apuntan al stub de Microsoft Store (no funciona)
- `py` → también puede fallar

---

## 📁 Estructura de carpetas (Dropbox)

**Raíz del proyecto:**
```
C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\
```
En bash/terminal: `/c/Users/User/Dropbox/Proyecto_Paisajes_Sonoros_Repositorio_Local/`

### Subcarpetas principales:

```
Proyecto_Paisajes_Sonoros_Repositorio_Local\
│
├── CLAUDE.md                          ← este archivo
├── .claude\
│   └── settings.json                  ← permisos de Claude Code para este proyecto
│
├── Campaña_Diciembre_2024\            ← carpeta principal de resultados organizados
│   ├── 01_Scripts\                    → 28 scripts Python del pipeline completo
│   ├── 02_Resultados_por_sitio\       → métricas, tablas, figuras por sitio (20 sitios)
│   ├── 03_Figuras_consolidadas\       → 318 figuras PNG/HTML consolidadas
│   │   ├── UMAP_2D\                   → 60 scatter plots UMAP por sitio
│   │   ├── UMAP_3D_interactivo\       → 20 HTMLs 3D interactivos (abrir en browser)
│   │   ├── PCA_varianza\              → 20 curvas de varianza explicada
│   │   ├── Ruido_vs_valido\           → 20 gráficos ruido HDBSCAN
│   │   ├── Subclustering\             → 178 figuras de subclustering
│   │   └── Overlay_BD\                → 40 figuras overlay BD anfibios (NO usar en congreso)
│   ├── 04_Metricas_globales\          → CSV con métricas de los 265 clusters
│   ├── 05_Figuras_Congreso_SOLABIMA2026\ ← FIGURAS FINALES PARA EL CONGRESO
│   │   ├── FigA_UMAP_Panel\           → UMAP scatter 3 hábitats (panel + individuales)
│   │   ├── FigB_Espectrogramas_Audio\ → Panel 2x2 espectrogramas + 4 WAVs para escuchar
│   │   ├── FigC_Subclustering\        → Antes/después subclustering PA-17 Cluster 0
│   │   ├── FigD_Violin_Habitats\      → Violin plots comparación por hábitat
│   │   └── scripts\                   → Scripts Python que generaron las figuras del congreso
│   └── RESUMEN_CAMPAÑA_DICIEMBRE_2024.md
│
├── resultados_HDD_Seagate\            ← resultados procesados (CSVs, parquets, figuras por sitio)
│   ├── Campaña diciembre 2024\        → una carpeta por sitio con todos los CSVs y figuras
│   ├── Fase1_Extracción_de_MFCCs...py
│   ├── Fase2_UMAP_HDBSCAN.py
│   └── ... (scripts del pipeline)
│
├── figuras_inspeccion_clusters\       ← grillas de espectrogramas por cluster (muestra_10 + representativos)
│   └── Campaña diciembre 2024\
│       └── [sitio]\
│           ├── [sitio]_cluster_N_muestra_10.png
│           └── [sitio]_cluster_N_representativos.png
│
├── validacion_matching_rep_vs_base\   ← comparación YAMNet entre cluster y BD anfibios
├── modelos_fase2_fase4\               ← modelos PCA/UMAP/HDBSCAN entrenados (.pkl, joblib)
├── matching_representativos_vs_base\  ← parquet con matching top-K de los 20 sitios
│   └── matching_representativos_vs_base_topK.parquet
│
├── BD_anfibios_wav\                   ← base de datos de referencia de anfibios en WAV
├── Pipeline_PDF_Campana_Principal\    ← pipeline original (4 sitios, PDF de resultados)
│   └── Literatura\                    → artículos científicos de referencia
│
└── Proyeccion_BD_en_Clusters\         ← scripts para proyectar BD en espacio UMAP
```

---

## 💾 Disco Duro Externo Seagate

**Acceso en Windows:** `E:\`
**Acceso en bash/Git Bash:** `/e/`

### Estructura en el Seagate:
```
E:\
└── Campaña diciembre 2024\
    ├── BO-31Tapyta\
    │   └── Data\
    │       └── BO-31TAPYTA_20241112_HHMMSS.wav   ← audios crudos (~1GB por sitio)
    ├── BO-32Tapyta\Data\*.wav
    ├── ... (20 sitios en total)
    └── PA-20Tapyta\Data\*.wav
```

### Reglas importantes para el Seagate:
- **NUNCA copiar los WAVs** al Dropbox — son demasiado pesados (varios GB por sitio)
- **Leer directamente desde el disco** cuando se necesitan los audios
- Para construir la ruta completa de un WAV: `E:\[sitio]\Data\[archivo].wav`
- En Python: `r'E:\Campaña diciembre 2024\PA-17Tapyta\Data\PA-17TAPYTA_20241130_215802.wav'`
- El campo `archivo_origen` en los CSVs tiene formato: `PA-17Tapyta/PA-17TAPYTA_20241130_215802.wav`
  → Traducción a ruta real: `E:\Campaña diciembre 2024\` + `[sitio]\Data\` + `[archivo].wav`

### Verificar si el Seagate está conectado:
```bash
ls /e/ 2>/dev/null | head -5 || echo "Seagate NO conectado"
```

---

## 🗺️ Sitios de muestreo — Campaña Nov–Dic 2024

**20 sitios** en 3 tipos de hábitat:

| Hábitat | Sitios | N | Color en figuras |
|---------|--------|---|-----------------|
| Bosque | BO-31, BO-32, BO-33, BO-34, BO-35, BO-36, BO-37, Bo-38, BO-39, BO-40 | 10 | `#2e7d32` (verde) |
| Eucaliptal | EU-16, EU-17, EU-18, EU-19, EU-20 | 5 | `#6a1b9a` (violeta) |
| Pastizal | PA-16, PA-17, PA-18, PA-19, PA-20 | 5 | `#e65100` (naranja) |

**Fechas reales de grabación:** 11 noviembre → 8 diciembre 2024
(La carpeta se llama "Campaña diciembre 2024" pero las grabaciones empezaron en noviembre)

**Filtro horario aplicado:** solo se analizan segmentos entre **18:00 y 06:00 h** (horario nocturno)

---

## ⚙️ Parámetros del pipeline (NO modificar sin acuerdo explícito)

### Fase 1 — Extracción de MFCCs:
- Ventana de audio: **4 segundos** con solapamiento de **2 segundos**
- MFCCs: **20 coeficientes** → 40 features por ventana (media + desviación estándar)
- Atenuación adaptativa: **−9 dB** en banda 6–12 kHz (reduce ruido de insectos)
- Total segmentos analizados: **~1,852,242** (promedio ~92,600 por sitio)

### Fase 2 — UMAP + HDBSCAN:
```
PCA:     umbral 95% varianza (adaptativo: 16–29 componentes según sitio), whiten=True
UMAP:    n_neighbors=60, min_dist=0.3, 3 dimensiones, métrica=coseno
HDBSCAN: min_cluster_size=800, min_samples=65, cluster_selection_epsilon=0.03, método=EOM
Scaler:  RobustScaler (antes de PCA)
```

### Sufijo estándar de archivos:
```
v2_horario18a06_insectos6a12
```
Todos los CSVs y figuras del pipeline llevan este sufijo en el nombre.

---

## 📊 Resultados globales — Campaña Nov–Dic 2024

- **265 clusters** totales (promedio 13.2 por sitio, rango 2–40)
- **Silhouette promedio:** 0.394 (rango 0.063–0.591)
- **Davies-Bouldin promedio:** 0.645
- **Ruido HDBSCAN promedio:** 6.8%

### Sitios destacados:
| Sitio | Hábitat | Clusters | Silhouette | Ruido |
|-------|---------|----------|-----------|-------|
| EU-16Tapyta | Eucaliptal | 2 | **0.591** (mejor) | 0.0% |
| Bo-38Tapyta | Bosque | ? | **0.544** | ~0% |
| PA-16Tapyta | Pastizal | 35 | 0.500 | ~18% |
| PA-17Tapyta | Pastizal | **40** (máximo) | 0.474 | 22.4% |
| BO-34 Tapyta | Bosque | ? | 0.063 (peor) | — |

### Por hábitat:
| Hábitat | Clusters prom. | Silhouette prom. | Ruido prom. |
|---------|---------------|-----------------|------------|
| Bosque | 8.1 | 0.368 | 3.1% |
| Eucaliptal | 3.6 | 0.404 | 0.0% |
| Pastizal | 33.2 | 0.437 | 20.9% |

---

## 🎭 Los 4 arquetipos acústicos identificados

Identificados visualmente en PA-17Tapyta. Usar esta nomenclatura en figuras:

| Arquetipo | Descripción | Cluster de referencia (PA-17) | Patrón en espectrograma |
|-----------|-------------|-------------------------------|------------------------|
| 🐸 Vocalización de anfibio | Llamados tónales repetitivos | Cluster 2 | Rayas verticales a 1–2 kHz |
| 🦗 Coro de insectos nocturnos | Estridulación broadband continua | Cluster 17 | Energía difusa broadband |
| 🌧️ Ruido abiótico | Lluvia — sin estructura tonal | Cluster 11 | Energía plana en todo el espectro |
| 🔀 Sonido mixto | Superposición de múltiples fuentes | Cluster 0 | Combinación de los anteriores |

---

## 🔬 Subclustering — cómo decide el pipeline

Un cluster es candidato al subclustering si cumple **las 3 condiciones**:
1. Tiene **≥ 1,000 segmentos**
2. `dist_media ≥ 0.8` (distancia promedio de segmentos al centroide en UMAP 3D)
3. `dist_p90 ≥ 1.2` (percentil 90 de esa distancia)

**Resultado:** 161 de 265 clusters refinados (66.8%). Ejemplo emblemático: Cluster 0 de PA-17 → 7 subclusters.

---

## 🎨 Especificaciones para figuras del congreso (SOLABIMA 2026)

### Parámetros técnicos obligatorios:
- **DPI:** 300
- **Fondo:** blanco (`facecolor='white'`)
- **Formato de salida:** PNG (y opcionalmente PDF)
- **Guardar en:** `05_Figuras_Congreso_SOLABIMA2026/[subcarpeta]/`

### Colores estándar del proyecto:
```python
COLORES_HABITAT = {
    'Bosque':     '#2e7d32',   # verde bosque
    'Eucaliptal': '#6a1b9a',   # violeta
    'Pastizal':   '#e65100',   # naranja tierra
}
NOISE_COLOR  = '#cccccc'       # gris para puntos de ruido HDBSCAN
```

### Estilo de ejes:
- Spines top y right: `set_visible(False)` (ocultos)
- Grid: líneas punteadas, alpha=0.4
- Fuente de título: negrita, tamaño 11–12
- Fuente de ejes: tamaño 9–10

### Figuras ya generadas (no regenerar sin motivo):
| Archivo | Descripción |
|---------|-------------|
| `FigA_UMAP_Panel_3Habitats.png` | Panel UMAP 3 hábitats lado a lado |
| `FigA_[sitio]_UMAP_mejorado.png` | UMAP individual con leyenda compacta |
| `FigB_Panel_Espectrogramas_4Arquetipos.png` | Panel 2×2 espectrogramas |
| `audio_Arquetipo_*.wav` | 4 clips de audio para escuchar en presentación |
| `FigC_Subclustering_AntesDespues.png` | Antes/después del subclustering PA-17 Cluster 0 |
| `FigD_Violin_Comparacion_Habitats.png` | Violin plots comparación por hábitat |

---

## 📝 Congreso SOLABIMA 2026

- **Abstract:** texto plano + LaTeX (sin tablas ni figuras en el abstract)
- **Formato:** según las instrucciones del sitio del congreso
- **Estado:** pipeline técnico completo ✅ | abstract y presentación ⬜ pendiente

### Lo que SÍ presentar:
- Pipeline MFCC + UMAP + HDBSCAN (los 20 sitios)
- Subclustering jerárquico automático
- Los 4 arquetipos acústicos
- Comparación entre hábitats (Figura D)

### Lo que NO presentar (decisión tomada):
- Overlay con la BD de anfibios
- Identificación de especies con YAMNet (matching con similitud coseno)
  Razón: el matching está sesgado por el coro de insectos de fondo, no es confiable

---

## 📌 Tareas pendientes al momento de la última sesión (mayo 2026)

- [ ] Umbral de confianza YAMNet: filtrar matches con similitud < 0.85
- [ ] Tabla consolidada: sitio × especie × n_detecciones × sim_media
- [ ] Figura de diversidad por hábitat adicional (si se necesita)
- [ ] Abstract para SOLABIMA 2026 (texto plano + LaTeX)
- [ ] Decidir poster vs presentación oral

---

## 🔑 Archivos más importantes para retomar el trabajo

| Archivo | Para qué sirve |
|---------|---------------|
| `Campaña_Diciembre_2024/RESUMEN_CAMPAÑA_DICIEMBRE_2024.md` | Resumen ejecutivo completo de toda la campaña |
| `Campaña_Diciembre_2024/pipeline_nodos_Seagate_Dic2024.html` | Grafo interactivo del pipeline (abrir en browser) |
| `Campaña_Diciembre_2024/04_Metricas_globales/*.csv` | Métricas de los 265 clusters (todos los sitios) |
| `resultados_HDD_Seagate/Campaña diciembre 2024/[sitio]/[sitio]_v2_..._fase2_umap_hdbscan.csv` | Coordenadas UMAP + cluster de cada segmento |
| `resultados_HDD_Seagate/Fase2_UMAP_HDBSCAN.py` | Script principal del pipeline |
| `matching_representativos_vs_base/matching_representativos_vs_base_topK.parquet` | Resultados del matching YAMNet (570 matches) |

---

## 🧩 Contexto técnico adicional

### Formato de archivo_origen en los CSVs:
```
PA-17Tapyta/PA-17TAPYTA_20241130_215802.wav
↓ se convierte en ruta real:
E:\Campaña diciembre 2024\PA-17Tapyta\Data\PA-17TAPYTA_20241130_215802.wav
```

### Los modelos entrenados (.pkl) están en:
```
modelos_fase2_fase4/Campaña diciembre 2024/modelos_[sitio]_v2_horario18a06_insectos6a12.pkl
```
Se cargan con `joblib.load()`, NO con `pickle.load()`.
Contienen: `sitio`, `scaler`, `pca`, `centroides_cluster`, `centroides_subcluster`.

### Comparación con el pipeline original (PDF, 4 sitios):
El pipeline original está en `Pipeline_PDF_Campana_Principal/` y usaba:
- PCA fijo 20 componentes (vs. adaptativo 95% varianza en Nov-Dic 2024)
- Atenuación −6 dB (vs. −9 dB en Nov-Dic 2024)
- Solo 4 sitios: BO-Tapyta, CANTERA1, EU-41, PA-41 (sitios distintos, no repetidos)
Los resultados de Nov-Dic 2024 son objetivamente mejores (Silhouette positivo en todos los sitios).
