# Proyecto Paisajes Sonoros — Bosque Atlántico del Alto Paraná, Tapytá, Paraguay
> Este archivo es leído automáticamente por Claude Code al inicio de cada sesión.
> Contiene todo el contexto necesario para retomar el trabajo sin perder tiempo.

---

## 🎯 ¿De qué trata este proyecto?

**Monitoreo Acústico Pasivo (PAM)** de anfibios en el Bosque Atlántico del Alto Paraná de Tapytá, Paraguay.

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

## 🔄 Protocolo de cierre de sesión (coordinación entre PC y notebook)

Este proyecto se trabaja desde dos máquinas (ver sección "Entornos Python"). Como el chat de Claude
Code **no se sincroniza** entre máquinas (cada una guarda su propio historial local), este archivo es
el único canal de coordinación que ambas cargan automáticamente.

**Regla fija: al cerrar cada sesión de trabajo (en cualquiera de las dos máquinas), actualizar la
sección "📌 Tareas pendientes" de este mismo archivo con un resumen breve de qué se hizo** (fecha +
2-3 líneas: qué se tocó, qué quedó a medias, qué decidir la próxima vez). Así la otra máquina lo ve
apenas Claude Code abra este archivo, sin depender de memoria ni de avisos manuales.

### ⚠️ Respaldo: rutina automática al INICIO de sesión (por si no se pudo anotar al cerrar)

El usuario suele cerrar la sesión de golpe (cerrar la terminal o apagar la PC directamente, sin avisar
que terminó) — en ese caso la nota de cierre de arriba **no llega a escribirse**. Como red de seguridad:

**Al iniciar cualquier sesión en este proyecto, antes de asumir que la última "Nota de sesión" de más
arriba refleja el estado real:**
1. Correr `git log -3 --oneline` y `git status --short` para ver si hay algo (commits, cambios sin
   commitear) que no esté mencionado en la nota más reciente.
2. Revisar fechas de modificación de archivos tocados recientemente (ej. `find <carpeta> -newermt
   "<fecha de la última nota>"`) para detectar trabajo que se hizo pero no quedó documentado.
3. Si aparece actividad no documentada, **reconstruir automáticamente qué se hizo** (a partir de esas
   fechas/commits, sin necesidad de que el usuario lo explique) y agregar una nota retroactiva en
   "Tareas pendientes → Notas de sesión" ANTES de seguir con el pedido del usuario — igual que se hizo
   el 2026-09-17 al reconstruir el trabajo del 11-13 de septiembre.
4. Si el usuario menciona algo que hizo y no coincide con lo que quedó escrito, priorizar lo que diga
   el usuario y corregir la nota.

---

## 💻 Entornos Python — REGLAS CRÍTICAS

Este proyecto se trabaja desde **dos máquinas** (PC de escritorio y notebook), cada una con su propio
usuario de Windows y por lo tanto su propia ruta de conda. El entorno `paisajes_matching` está
replicado (mismos paquetes, ver `requirements.txt`) en ambas. **A partir de julio 2026 el trabajo
principal se hace desde la notebook** (uso de la PC de escritorio muy reducido).

| Máquina | Usuario Windows | Ruta del entorno |
|---------|------------------|-------------------|
| PC de escritorio | `User` | `/c/Users/User/miniconda3/envs/paisajes_matching/python` |
| Notebook | `Alcides` | `/c/Users/Alcides/miniconda3/envs/paisajes_matching/python` |

**Antes de correr un script, detectar en qué máquina se está** (`whoami` o `echo $USERNAME`) y usar
la ruta correspondiente.

**Cómo correr scripts (ejemplo notebook):**
```bash
PYTHONUTF8=1 /c/Users/Alcides/miniconda3/envs/paisajes_matching/python nombre_script.py
```
**Cómo correr código inline (ejemplo notebook):**
```bash
PYTHONUTF8=1 /c/Users/Alcides/miniconda3/envs/paisajes_matching/python -c "..."
```
El prefijo `PYTHONUTF8=1` es **obligatorio** en Windows para evitar errores con tildes y ñ.

### Entorno base (solo para PDF con pypdf/pdfplumber):
```
PC de escritorio:  /c/Users/User/miniconda3/python
Notebook:          /c/Users/Alcides/miniconda3/python
```

### ⚠️ Nunca usar:
- `python3` o `python` solos → stub de Microsoft Store
- `py` → puede fallar

### 💾 Disco duro externo Seagate
Es un disco físico USB — solo está disponible en la máquina a la que esté conectado en ese momento
(ver sección "Disco Duro Externo Seagate" más abajo). Si `/e/` no existe, primero revisar que el
disco esté enchufado a la máquina actual antes de asumir un error.

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

### 📝 Notas de sesión (más reciente primero)

**2026-09-21 (PC de escritorio), cierre de sesión:** El usuario ya está editando
`presentacion_solabima2026.tex` **directamente en Overleaf** (no en este archivo local) — mi copia
local la mantengo espejada con lo que se fue pegando en Overleaf, pero **verificar cuál es más
reciente antes de asumir nada** la próxima vez. Estado al cerrar (17 diapositivas, ya compiló bien):
- Autores completos (15, Alcides Rojas resaltado) con afiliaciones en texto corrido compacto
  (`$^n$` + `\quad`/`\\`, NO `\inst{}` en el `\institute{}` porque genera lista vertical en Beamer).
- Diapositiva de cierre = portada con `\subtitle{¡GRACIAS!}` en vez de la descripción de la PoC.
  `\date{}` corregido a "Noviembre 2026" (mes real de SOLABIMA 2026, confirmado por el usuario).
- **Logos ya resueltos y activados:** CONACYT, PROCIENCIA, FP-UNA (convertido de SVG a PDF con
  `svglib`/`reportlab`, sin necesitar Cairo), Fundación Moisés Bertoni, SOLABIMA 2026. Archivos en
  `07_Presentacion_SOLABIMA2026/logos/` (los que se usan) y `logos_raw/` (alternativas descartadas,
  por si se quiere cambiar — ojo, esta carpeta tiene `.gif`/`.webp` que **no están cubiertos por el
  `.gitignore` actual**, no hacer `git add` amplio ahí sin revisar). Tamaño `height=0.7cm` en fila de
  5 para que entren sin desbordar.
- Diapositiva "Diversidad acústica" se dividió en dos (metodología / resultados) porque una sola no
  entraba — ya con los valores reales calculados (ver nota anterior) y prueba de permutación.
- Nada quedó pendiente explícito de esta sesión — el usuario va a seguir en otra sesión, probablemente
  revisando detalles visuales tras compilar en Overleaf.

**2026-09-21 (PC de escritorio):** Se calculó $R_s$/$H_s$/$D_s$ real (Riqueza, Entropía Shannon,
Dominancia Simpson) por sitio y la prueba de permutación (10.000 réplicas), pendiente desde el
20-sept. Nuevo script verificado: `01_Scripts/Fase5_Diversidad_Riqueza_Entropia_Dominancia.py`.
**Lógica de reconstrucción del cluster final:** tipo = `(cluster_hdbscan, subcluster_id)` si
`subcluster_id>=0`, si no `(cluster_hdbscan, -1)`; excluyendo ruido HDBSCAN (`cluster_hdbscan==-1`).
**Verificado: la suma de $R_s$ en los 20 sitios da exactamente 265** (coincide con lo publicado) —
confirma que la reconstrucción es correcta antes de confiar en $H_s$/$D_s$/permutación.
Resultado: diferencias entre hábitats significativas en las 3 métricas
(Pastizal $\overline{R_s}=28{,}6$, $\overline{H_s}=2{,}76$, $\overline{D_s}=0{,}189$;
Bosque 9,8 / 1,45 / 0,378; Eucaliptal 4,8 / 0,67 / 0,669;
permutación: $R_s\,p=0{,}0037$, $H_s\,p=0{,}0072$, $D_s\,p=0{,}0229$). Ya actualizado en la
diapositiva "Diversidad acústica" de `presentacion_solabima2026.tex` (reemplazó la nota de
"pendiente"). **Pendiente:** decidir si estos resultados también se agregan al documento técnico/
personal (ahí no están, solo en el resumen como metodología planeada y ahora en la presentación).

También en la presentación (mismo día): lista completa de 15 autores (Alcides Rojas resaltado,
corregido typo "Roja"→"Rojas") en portada; diapositiva final ahora es una copia de la portada +
"¡GRACIAS!". El usuario va a seguir editando **directamente en Overleaf** de acá en más (no en este
archivo local) — verificar antes de asumir que el `.tex` local sigue siendo la versión más reciente.

**2026-09-20 (PC de escritorio), cierre de sesión:** Presentación
`presentacion_solabima2026.tex` ya **subida a Overleaf y compartida con Pastor** para revisión. Estado
al cerrar:
- 16 diapositivas: contexto, objetivo, área de estudio, datos (grabadora corregida), pipeline,
  subclustering, diversidad acústica (marco metodológico $R_s$/$H_s$/permutación, sin valores
  inventados), resultados globales, UMAP, caso PA-17 (con cita Thomas et al. 2022), verificación de
  consistencia (vacía a propósito — Fig 5/8 del doc. personal), hallazgo temporal, comparación
  espectral, conclusiones, agradecimientos (CONACYT + ARASY con nombre completo confirmado), cierre.
- **Pendiente para el usuario (no bloqueante):** subir archivos de logo reales (CONACYT, FP-UNA/NIDTEC,
  Fundación Moisés Bertoni, SOLABIMA) a una carpeta `logos/` en Overleaf y descomentar las líneas
  `\includegraphics` marcadas `% TODO` en portada y agradecimientos.
- **Pendiente para otra sesión:** (1) decidir si mejorar/incluir Fig 5 y Fig 8 (grillas de
  espectrogramas) o descartarlas — diapositiva placeholder ya lista para completar o eliminar;
  (2) calcular $H_s$/$D_s$ reales y la prueba de permutación (ver nota anterior — requiere leer
  `Fase_4_2_Subclustering_jerarquico_automático_dentro_de_clusters.py` para reconstruir el ID de
  cluster final antes de calcular, y verificar que dé 265 clusters en total).
- Nada del pipeline/análisis de datos se tocó hoy — todo el trabajo fue de la presentación.

**2026-09-20 (PC de escritorio), continuación — presentación SOLABIMA:** Se creó
`Campaña_Diciembre_2024/07_Presentacion_SOLABIMA2026/presentacion_solabima2026.tex` (Beamer, tema
Madrid/seahorse) con una presentación oral tentativa, basada en el resumen enviado a SOLABIMA 2026 y
en `documento_tecnico_paisajes_sonoros.tex` (para datos corregidos). **Ojo:** el resumen enviado dice
grabadora "AudioMoth" — dato desactualizado; la presentación usa el dato correcto (Wildlife Acoustics
Song Meter Micro 2) del documento técnico. Autores/afiliaciones quedaron como placeholder editable a
propósito (no se copiaron del resumen). Figuras copiadas a `07_Presentacion_SOLABIMA2026/figs/` desde
`06_Documento_Tecnico/figs_overleaf/` (mismas imágenes comprimidas ya usadas en el doc técnico).
Sigue en revisión — es tentativa, se va a seguir ajustando diapositivas.

**Ajuste posterior (mismo día):** se identificó que "Figura 5" y "Figura 8" del documento personal
(`FigF1_Centroides_7subclusters` y `FigG1_Centroides_8clusters` — grillas con muchos espectrogramas,
poca diferencia visible) no debían incluirse todavía. Se agregó una diapositiva "Verificación de
consistencia (en revisión)" con el texto pero sin las imágenes (comentadas en el `.tex`, listas para
descomentar cuando se decida). También se agregó cita a Thomas et al. (2022) que faltaba en la
diapositiva de arquetipos PA-17 (el resumen SOLABIMA la menciona explícitamente).
**Resuelto:** el usuario confirmó agregar las métricas formales de diversidad ($R_s$, $H_s$,
permutación) y confirmó que Fig 5/8 son del documento personal. Se agregó la diapositiva "Diversidad
acústica: marco metodológico" con las fórmulas. **Importante — hallazgo durante esto:** los valores
reales de Entropía ($H_s$) y Dominancia ($D_s$) por sitio, y el resultado de la prueba de permutación
de 10.000 réplicas, **no están calculados todavía** — confirmado que sigue como pendiente explícito en
`RESUMEN_CAMPAÑA_DICIEMBRE_2024.md` ("Figura de diversidad por hábitat"). Los datos crudos para
calcularlo existen (`resultados_HDD_Seagate/Campaña diciembre 2024/[sitio]/..._fase4_umap_hdbscan_
subclusters.csv`, ~100k segmentos/sitio con `cluster_hdbscan` + `subcluster_id`), pero reconstruir el
ID de cluster final (base+subcluster) que reconcilie exactamente con los "265 clusters" ya publicados
requiere un análisis dedicado (no se improvisó para no arriesgar un número que contradiga lo ya
publicado). **Pendiente real para otra sesión:** calcular $H_s$/$D_s$ y la permutación como script
aparte, verificando que el total de clusters resultante siga dando 265 antes de reportar cualquier
número nuevo.

**ARASY confirmado (vía web):** nombre completo "ARASY: Potenciamiento de las capacidades de
modelamiento matemático computacional para la tierra y la vida" (ESTR01-23), financiado por CONACYT
vía PROCIENCIA II/FEEI, ejecutado por FP-UNA/NIDTEC. **No tiene logo propio** (confirmado por el
usuario y por conacyt.gov.py/arasy) — se usa el logo de CONACYT + FP-UNA/NIDTEC en su lugar. Ya
actualizado en la diapositiva de Agradecimientos del `.tex`. Sigue pendiente: el usuario debe subir
los archivos de logo reales a una carpeta `logos/` para descomentar esas líneas.

**Corrección (2026-09-21):** "PINV" NO es "Programa Nacional de Incentivo a la Investigación" (dato
erróneo que yo mismo había puesto) — es el código de convocatoria "Proyectos de Investigación" dentro
del **Programa PROCIENCIA** de CONACYT (fondos FEEI), el mismo programa marco que financia ARASY
(código ESTR = "Proyectos Estratégicos"). **PRONII es un programa totalmente distinto** (categoriza/
incentiva a investigadores individuales, no financia proyectos) — no corresponde usar su logo acá.
Logos correctos para PINV01-530 y ARASY ESTR01-23: **CONACYT + PROCIENCIA** (no PRONII). Ya corregido
el texto de Agradecimientos y los comentarios de logos en el `.tex` local — falta reflejar el cambio
en Overleaf (usuario edita ahí directamente ahora).

**2026-09-20 (PC de escritorio):** Se resolvió lo pendiente del 17 de sept:
1. Commit `baae411` (CLAUDE.md, .gitignore, settings.json, INDICE_PAPERS.md) commiteado y pusheado.
2. Se instaló `git-filter-repo` y se purgó de **toda la historia de git** (no de Dropbox — ningún
   archivo de la carpeta del proyecto se tocó) el contenido de `Datos/`, `Salida/`, `figuras/`,
   `figuras_PA-41Tapyta/`, `figuras_inspeccion_clusters/`, `resultados/`,
   `selection_tables_por_cluster*/` y `.gitignore.txt` — los `.wav`/`.txt`/`.png` de mayo 2026 que
   quedaron expuestos en el repo público de GitHub antes de que existiera el `.gitignore` actual.
   `requirements.txt` y todo el código se preservaron intactos (verificado).
3. `.git` bajó de 701 MB a **1.3 MB**. Se hizo `git push origin main --force` — GitHub ya sirve la
   historia limpia (commit actual: `6748689`). **Importante:** los hashes de todos los commits
   cambiaron por la reescritura; si alguna otra copia local (ej. la notebook) tiene un clon viejo del
   repo, va a divergir — conviene que en la notebook se haga `git fetch origin && git reset --hard
   origin/main` (sin cambios locales pendientes ahí) en vez de un `pull` normal.
4. El backup completo de la historia vieja (con los archivos grandes, por si hiciera falta recuperar
   algo) sigue disponible en `D:\Backups_Git\Proyecto_Paisajes_Sonoros\` en esta PC de escritorio
   (fuera de Dropbox, no se sincroniza).

**2026-09-17 (PC de escritorio):** Sesión de verificación de sincronización PC↔notebook (sin cambios
al pipeline). Se confirmó que `documento_tecnico_paisajes_sonoros.tex/pdf` y
`guia_personal_paisajes_sonoros.tex/pdf` (actualizados 11-13 sept, probablemente desde la notebook)
llegaron bien vía Dropbox.

Se investigó el estado de git y se hicieron dos cosas:
- **Se identificaron los archivos grandes atrapados en la historia de git** (commits de mayo 2026,
  antes del `.gitignore` actual): ~1026 MB en 2322 blobs — `.wav` 652.3 MB (1922 archivos, audio que
  nunca debió commitearse), `.txt` 292.8 MB (124 archivos, tablas `selection_tables_por_cluster...`
  de Raven), `.png` 75.9 MB (152 archivos, figuras de resultados). El resto (`.py`/`.md`/etc., ~5 MB)
  sí corresponde al repo.
- **Se movió `.git_backup_pre_purge_20260822_212253/` (2.4 GB, copia de seguridad de un intento de
  purga del 22 ago que nunca se completó) fuera de Dropbox**, a `D:\Backups_Git\Proyecto_Paisajes_Sonoros\`
  en la PC de escritorio (disco "Almacenamiento EX (D)"). Verificado con robocopy: 5755 archivos,
  2.4 GB, 0 errores, copia idéntica antes de borrar el original. Esto liberó 2.4 GB de la carpeta
  sincronizada — **la copia de este backup NO está en Dropbox, solo existe en el disco D: de la PC de
  escritorio.**

**PENDIENTE (a propósito, no se tocó todavía — se decidió continuar en otra sesión, posiblemente
desde la notebook):**
1. El repo de git sigue en el commit del 6 jun 2026 (`25130b8`, igual que `origin/main` en GitHub).
   Hay cambios sin commitear en `CLAUDE.md`, `.gitignore` y `.claude/settings.json` (ediciones del
   9 jul nunca guardadas), más `.claude/settings.local.json` y `Antecedentes_Literatura_Cientifica/`
   sin trackear. **Resolver esto primero**, antes de tocar la historia de git.
2. Terminar la purga real de historia con `git filter-repo` (o BFG) para sacar los `.wav`/`.txt`/`.png`
   listados arriba (~1 GB) — reduciría el `.git` de 701 MB a probablemente <50 MB. No hacerlo con las
   dos máquinas escribiendo al mismo repo al mismo tiempo (el `.git` vive dentro de Dropbox — riesgo
   de corromperlo si se usa desde ambas casi al mismo tiempo). Hacer `git fetch`/`pull` primero en la
   máquina donde se retome esto para partir de un estado limpio.

**2026-09-11 a 13 (notebook, inferido por timestamps):** Se revisó/actualizó
`guia_personal_paisajes_sonoros` (con subrayados, ver backup `..._CON_SUBRAYADOS_backup.pdf`) y
`documento_tecnico_paisajes_sonoros`. No se generó commit de git (son archivos ignorados a propósito,
ver `.gitignore`).

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
