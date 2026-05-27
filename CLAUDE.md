# Proyecto Paisajes Sonoros — Bosque Atlántico Interior, Tapytá, Paraguay
> Este archivo es leído automáticamente por Claude Code al inicio de cada sesión.
> Contiene todo el contexto necesario para retomar el trabajo sin perder tiempo.

---

## 🎯 ¿De qué trata este proyecto?

**Monitoreo Acústico Pasivo (PAM)** de anfibios en el Bosque Atlántico Interior de Tapytá, Paraguay.

El pipeline analiza grabaciones nocturnas registradas con grabadoras autónomas (AudioMoth), extrae características acústicas (MFCCs), aplica reducción dimensional (UMAP) y agrupamiento automático (HDBSCAN) para identificar y clasificar tipos de sonidos sin supervisión humana.

**Objetivo principal:** Presentar resultados en el congreso **SOLABIMA 2026** (Sociedad Latinoamericana de Biología Matemática).

**Contexto ecológico:** Grabaciones al inicio de la temporada húmeda subtropical (noviembre–diciembre), que coincide con el pico de actividad vocal de los anfibios.

**Repositorio GitHub (público):** https://github.com/AlcidesArielRojas/analisis_bioacustico_anfibios

---

## 👤 Usuario y preferencias generales

- Comunicación: **español**, en tono claro y accesible
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
El prefijo `PYTHONUTF8=1` es **obligatorio** en Windows para evitar errores con tildes y ñ.

### Entorno base (solo para PDF con pypdf/pdfplumber):
```
Ruta:  /c/Users/User/miniconda3/python
```

### ⚠️ Nunca usar:
- `python3` o `python` solos → stub de Microsoft Store
- `py` → puede fallar

---

## 📁 Estructura de carpetas (Dropbox)

**Raíz del proyecto:**
```
C:\Users\User\Dropbox\Proyecto_Paisajes_Sonoros_Repositorio_Local\
```

```
Proyecto_Paisajes_Sonoros_Repositorio_Local\
│
├── CLAUDE.md                              ← este archivo
├── .claude\settings.json                  ← permisos Claude Code para este proyecto
├── README.md                              ← documentación pública (GitHub)
│
├── Campaña_Diciembre_2024\
│   ├── 01_Scripts\                        → 28 scripts Python del pipeline
│   ├── 02_Resultados_por_sitio\           → métricas y figuras por sitio (20 sitios)
│   ├── 03_Figuras_consolidadas\           → 318 figuras PNG/HTML
│   ├── 04_Metricas_globales\              → CSV con métricas de los 265 clusters
│   ├── 05_Figuras_Congreso_SOLABIMA2026\  ← FIGURAS FINALES
│   │   ├── FigA_UMAP_Panel\
│   │   ├── FigB_Espectrogramas_Audio\
│   │   ├── FigC_Subclustering\
│   │   ├── FigD_Violin_Habitats\
│   │   ├── FigE_Huellas_Espectrales\
│   │   ├── FigF_Verificacion_Subclustering\
│   │   ├── FigG_Grillas_Consistencia\
│   │   ├── FigH_Exploracion_Vocalizaciones\
│   │   ├── FigI_Busqueda_Aves\
│   │   ├── FigJ_Actividad_Temporal\
│   │   ├── FigK_Comparacion_Espectral_Anfibios\
│   │   └── scripts\                       → Script_E a Script_K.py
│   └── 06_Documento_Tecnico\              ← DOCUMENTACIÓN LATEX
│       ├── documento_tecnico_paisajes_sonoros.tex  → informe formal ~13 páginas
│       ├── guia_personal_paisajes_sonoros.tex      → guía didáctica ~38 páginas
│       ├── referencias.bib                         → 13 referencias BibTeX
│       ├── figs\                                   → 15 PNG originales 300 DPI
│       └── figs_overleaf\                          → 15 JPEG comprimidos (8 MB total)
│                                                     usar estos para Overleaf gratuito
│
├── Antecedentes_Literatura_Cientifica\    ← 3 papers open access (mayo 2026)
│   ├── Thomas_et_al_2022_Unsupervised_UMAP_HDBSCAN_Vocalizaciones_JAE.pdf
│   ├── Villanueva-Rivera_2014_Acoustic_Niche_Partitioning_Anurans_PeerJ.pdf
│   ├── Donnelly_et_al_2026_Acoustic_Niche_Atlantic_Forest_Conservation.pdf
│   └── INDICE_PAPERS.md
│
├── Pipeline_PDF_Campana_Principal\
│   └── Literatura\                        → 9 papers metodológicos (sesión anterior)
│
├── resultados_HDD_Seagate\
├── figuras_inspeccion_clusters\
├── modelos_fase2_fase4\
├── matching_representativos_vs_base\
├── BD_anfibios_wav\
└── Proyeccion_BD_en_Clusters\
```

---

## 💾 Disco Duro Externo Seagate

**Windows:** `E:\` · **Git Bash:** `/e/`

```
E:\Campaña diciembre 2024\[sitio]\Data\[sitio]_YYYYMMDD_HHMMSS.wav
```

- **NUNCA** copiar WAVs al Dropbox
- Leer directamente desde `E:\` cuando se necesitan audios
- `archivo_origen` en CSV → `PA-17Tapyta/PA-17TAPYTA_20241130_215802.wav`
  → ruta real: `E:\Campaña diciembre 2024\PA-17Tapyta\Data\PA-17TAPYTA_20241130_215802.wav`

```bash
ls /e/ 2>/dev/null | head -5 || echo "Seagate NO conectado"
```

---

## 🗺️ Sitios de muestreo — Campaña Nov–Dic 2024

| Hábitat | Sitios | N | Color |
|---------|--------|---|-------|
| Bosque | BO-31 … BO-40 | 10 | `#2e7d32` verde |
| Eucaliptal | EU-16 … EU-20 | 5 | `#6a1b9a` violeta |
| Pastizal | PA-16 … PA-20 | 5 | `#e65100` naranja |

Grabación: **11 nov → 8 dic 2024** · Filtro horario: **18:00–06:00 h**

---

## ⚙️ Parámetros del pipeline (NO modificar sin acuerdo explícito)

### Fase 1 — Extracción de MFCCs:
- Ventana: **4 s**, solapamiento **2 s**
- **20 MFCC** → 40 features/ventana (media + DE)
- Atenuación adaptativa: **−9 dB** en 6–12 kHz
- Total segmentos: **~1,852,242** (~92,600/sitio)

### Fase 2 — UMAP + HDBSCAN:
```
PCA:     95% varianza, whiten=True (16–29 componentes)
UMAP:    n_neighbors=60, min_dist=0.3, n_components=3, metric=coseno
HDBSCAN: min_cluster_size=800, min_samples=65, epsilon=0.03, method=EOM
Scaler:  RobustScaler
Sufijo:  v2_horario18a06_insectos6a12
```

---

## 📊 Resultados globales

- **265 clusters** · silueta 0.394 · Davies-Bouldin 0.645 · ruido 6.8%
- Pastizal: 33.2 clusters/sitio · Bosque: 8.1 · Eucaliptal: 3.6
- Subclustering: 161/265 clusters refinados (66.8%)

---

## 🔬 Hallazgos clave — PA-17 (Pastizal)

### Arquetipos acústicos confirmados (inspección visual + auditiva):
| Cluster | Categoría | N segs |
|---------|-----------|--------|
| Cl.2 | Vocalización anfibio (ancla) | 1,454 |
| Cl.1 | Vocalización anfibio | 1,450 |
| Cl.3 | Vocalización anfibio | 5,809 |
| Cl.39 | Canto de ave (ancla) | 3,528 |
| Cl.17 | Coro de insectos | 10,884 |
| Cl.38 | Lluvia / tormenta | 1,413 |
| Cl.0 | Mixto (vocal + insectos) | 953 |

### Actividad temporal nocturna (18:00–06:00 h):
| Categoría | N segs | % | Hora pico |
|-----------|--------|---|-----------|
| Insectos | 18,110 | 55.4% | 03:00 |
| Anfibios | 8,713 | 26.6% | 00:00 |
| Aves | 3,528 | 10.8% | 18:00 |
| Lluvia | 1,413 | 4.3% | 18:00 |
| Mixto | 953 | 2.9% | 19:00 |

### Partición temporal del nicho acústico (hallazgo principal):
Los 3 clusters de anfibio son **espectralmente idénticos** pero **temporalmente segregados**:
- Cl.1 → pico **18:00 h**
- Cl.2 → pico **21:00 h**
- Cl.3 → pico **00:00 h**
- Frecuencia dominante: **~2,850 Hz** (rango: 2,832–2,907 Hz)
- Ancho de banda a −10 dB: **~1,400–3,200 Hz**

---

## 🎨 Figuras para el congreso — todas generadas ✅

| Fig | Script | Descripción |
|-----|--------|-------------|
| FigA | Script_A | Panel UMAP 3 hábitats (PA-17, EU-16, BO-31) |
| FigB | Script_B | Panel 2×2 espectrogramas 4 arquetipos + 4 WAVs |
| FigC | Script_C | Subclustering Cl.0 PA-17: antes/después (7 sub-clusters) |
| FigD | Script_D | Violin plots comparación por hábitat |
| FigE | Script_E | Huellas espectrales (PSD) 4 arquetipos ±1 DE |
| FigF | Script_F | Grillas 7 sub-clusters (centroide/aleatorio/extremo) |
| FigG | Script_G | Grillas consistencia inter-sitio (PA-17, EU-16, BO-31) |
| FigH | Script_H | UMAP 2D PA-17 — clusters candidatos vocalizaciones |
| FigI | Script_I | UMAP 2D PA-17 — búsqueda clusters de aves |
| FigJ | Script_J | Actividad temporal nocturna (3 paneles) |
| FigK | Script_K | Comparación espectral 3 clusters anfibio (4 paneles) |

**Colores estándar:**
```python
COLORES_HABITAT = {'Bosque': '#2e7d32', 'Eucaliptal': '#6a1b9a', 'Pastizal': '#e65100'}
NOISE_COLOR = '#cccccc'
```

---

## 📄 Documentación técnica (Overleaf)

Dos proyectos Overleaf separados compilados en PDF:

**Proyecto 1 — Documento técnico formal** (~13 páginas, estilo CS directo):
- `documento_tecnico_paisajes_sonoros.tex`
- Secciones: Resumen → Datos → Metodología → Resultados → Entorno computacional → Conclusiones

**Proyecto 2 — Guía personal didáctica** (~38 páginas, lenguaje cotidiano):
- `guia_personal_paisajes_sonoros.tex`
- Usa `tcolorbox` (azul=concepto, verde=hallazgo, naranja=importante, gris=nota técnica)
- No tocar sin motivo explícito

**IMPORTANTE para Overleaf (plan gratuito):**
- Subir figuras desde `figs_overleaf/` (JPEG, 8 MB total) — NO los PNG de `figs/` (27 MB)
- El `.tex` ya apunta a `.jpg` — no cambiar extensiones
- `referencias.bib` va en la raíz del proyecto Overleaf

---

## 🐙 GitHub

```
URL:  https://github.com/AlcidesArielRojas/analisis_bioacustico_anfibios
Rama: main   |   Visibilidad: pública
```

`.gitignore` excluye: `*.wav`, `*.csv`, `*.parquet`, `*.png`, `*.pkl`
→ Solo código fuente y documentación están en el repo.

`git push` funciona con Windows Credential Manager (no requiere configuración extra).

---

## 📚 Literatura científica

### `Antecedentes_Literatura_Cientifica/` — perspectiva ecológica (mayo 2026)
| Paper | Relevancia |
|-------|-----------|
| Thomas et al. 2022 (J. Animal Ecology) | Blueprint metodológico UMAP+HDBSCAN para vocalizaciones |
| Villanueva-Rivera 2014 (PeerJ) | Hipótesis nicho acústico en anuros — contrapunto a nuestro hallazgo |
| Donnelly et al. 2026 (Conservation) | Partición nicho acústico en comunidad anura del Bosque Atlántico |

### `Pipeline_PDF_Campana_Principal/Literatura/` — perspectiva metodológica (sesión anterior)
Best 2023, Sainburg 2020, Schneider 2022, Guerrero 2023, Alexander 2025,
Xu 2025, Canas 2023, Frasier 2021, Aide 2013.

---

## 📌 Tareas pendientes

### Para el congreso SOLABIMA 2026:
- [ ] **Abstract** (texto plano + LaTeX, sin tablas ni figuras)
- [ ] **Decidir formato:** presentación oral vs. póster

### Para el futuro (no urgente):
- [ ] **Web personal / portfolio** — GitHub Pages con plantilla `al-folio`
  Proyectos a destacar: Paisajes Sonoros (flagship), Micrurus, Topa Dengue
- [ ] **GitHub Pages** para este repositorio (visualización del pipeline)
- [ ] **"Anfibios de la Reserva Natural Tapytá.pdf"** — leer e identificar
  si los 3 clusters anfibio corresponden a especies conocidas del sitio

### Descartado (decisión tomada):
- ~~YAMNet / matching con BD de anfibios~~ — sesgado por coro de insectos

---

## 🔑 Archivos clave para retomar trabajo

| Archivo | Para qué |
|---------|---------|
| `Campaña_Diciembre_2024/RESUMEN_CAMPAÑA_DICIEMBRE_2024.md` | Resumen ejecutivo |
| `Campaña_Diciembre_2024/04_Metricas_globales/*.csv` | Métricas 265 clusters |
| `resultados_HDD_Seagate/Campaña diciembre 2024/[sitio]/..._fase2_umap_hdbscan.csv` | UMAP + clusters por segmento |
| `06_Documento_Tecnico/documento_tecnico_paisajes_sonoros.tex` | Informe formal |
| `06_Documento_Tecnico/guia_personal_paisajes_sonoros.tex` | Guía personal |
| `Antecedentes_Literatura_Cientifica/INDICE_PAPERS.md` | Índice de papers |

---

## 🧩 Notas técnicas adicionales

- Modelos `.pkl` → `joblib.load()` (NO pickle). Contienen: scaler, pca, centroides.
- Pipeline original (4 sitios, `Pipeline_PDF_Campana_Principal/`): PCA fijo 20 comp., −6 dB — resultados inferiores.
- `figuras_inspeccion_subclusters/` → PDFs de resumen de subclusters por sitio (20 archivos).
- `2503.15074v1.pdf` en raíz del proyecto → preprint arXiv sin identificar (revisar).
