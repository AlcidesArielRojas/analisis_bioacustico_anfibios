# Pipeline de Análisis Acústico — Paisajes Sonoros Tapytá

**Monitoreo Acústico Pasivo (PAM) de anfibios en el Bosque Atlántico del Alto Paraná**
Reserva Natural Tapytá, Caazapá, Paraguay — Campaña noviembre–diciembre 2024

---

## ¿De qué trata este proyecto?

Este repositorio contiene el pipeline completo de análisis bioacústico aplicado a
grabaciones nocturnas de 20 sitios de muestreo distribuidos en tres tipos de hábitat:
**bosque nativo**, **eucaliptal** y **pastizal**. El flujo de trabajo combina extracción
de descriptores acústicos (MFCC), reducción de dimensionalidad (PCA + UMAP) y
agrupamiento no supervisado (HDBSCAN) para identificar y caracterizar automáticamente
los tipos de sonidos presentes en el paisaje sonoro nocturno, con énfasis en la
detección de vocalizaciones de anfibios.

Los resultados de esta campaña serán presentados en el congreso
**SOLABIMA 2026** (Sociedad Latinoamericana de Biología Matemática).

---

## Hallazgos principales

- **265 clusters** detectados en 20 sitios (promedio 13,2 por sitio; rango 2–40)
- **Coeficiente de silueta global:** 0,394 — calidad de agrupamiento positiva en los 20 sitios
- El **pastizal** presenta la mayor complejidad acústica (33,2 clusters/sitio), el **eucaliptal** la menor (3,6 clusters/sitio)
- En PA-17 se identificaron **3 clusters de vocalización de anfibio** con frecuencia dominante convergente (~2.850 Hz) y **segregación temporal** escalonada: picos de actividad a las 18:00, 21:00 y 00:00 h — posible partición del nicho acústico nocturno

---

## Estructura del repositorio

```
📁 Campaña_Diciembre_2024/
│
├── 01_Scripts/                  → Scripts del pipeline principal (Fases 1 y 2)
│
└── 05_Figuras_Congreso_SOLABIMA2026/
    └── scripts/                 → Scripts de análisis y figuras para el congreso
        ├── Script_E_Huellas_Espectrales_PA17.py
        ├── Script_F_Verificacion_Subclustering.py
        ├── Script_G_Grillas_Consistencia_Clusters.py
        ├── Script_H_Exploracion_Vocalizaciones_PA17.py
        ├── Script_I_Busqueda_Aves_PA17.py
        ├── Script_J_Actividad_Temporal_PA17.py
        └── Script_K_Comparacion_Espectral_Anfibios_PA17.py

📁 Campaña_Diciembre_2024/06_Documento_Tecnico/
    ├── documento_tecnico_paisajes_sonoros.tex   → Documento formal para investigadores
    ├── guia_personal_paisajes_sonoros.tex       → Guía didáctica del pipeline
    └── referencias.bib                          → Referencias bibliográficas (BibTeX)

📄 CLAUDE.md                     → Documentación completa del proyecto para Claude Code
```

---

## Pipeline — Resumen de pasos

| Fase | Descripción | Script(s) |
|------|-------------|-----------|
| **1** | Filtro horario (18–06 h), atenuación de insectos (−9 dB, 6–12 kHz), segmentación en ventanas de 4 s, extracción de 20 MFCC → 40 features por segmento | `01_Scripts/` |
| **2** | RobustScaler → PCA (95% varianza, whitening) → UMAP 3D → HDBSCAN | `01_Scripts/` |
| **E** | Huellas espectrales (espectros de potencia media) de los 4 arquetipos acústicos | `Script_E` |
| **F** | Verificación de subclustering del Cluster 0 de PA-17 (7 sub-clusters) | `Script_F` |
| **G** | Consistencia de clusters entre sitios y hábitats | `Script_G` |
| **H** | Exploración de clusters con vocalizaciones adicionales (UMAP 2D + audio WAV) | `Script_H` |
| **I** | Búsqueda de clusters de aves usando ancla acústica | `Script_I` |
| **J** | Actividad temporal nocturna por categoría acústica (18:00–06:00 h) | `Script_J` |
| **K** | Comparación espectral de los 3 clusters de anfibio de PA-17 | `Script_K` |

---

## Entorno y dependencias

**Python 3.11.14** · Anaconda distribution · Windows 10

| Biblioteca | Versión | Uso principal |
|-----------|---------|--------------|
| `librosa` | 0.11.0 | Lectura de audio y cálculo de MFCC |
| `numpy` | 2.4.2 | Operaciones matriciales |
| `pandas` | 3.0.1 | Manipulación de datos |
| `umap-learn` | 0.5.11 | Reducción dimensional |
| `hdbscan` | 0.8.41 | Clustering |
| `scikit-learn` | 1.8.0 | PCA, escalado, métricas |
| `matplotlib` | 3.10.8 | Visualización |
| `soundfile` | 0.13.1 | Exportación de audio WAV |
| `scipy` | 1.17.1 | Procesamiento de señal |

---

## Sitios de muestreo

| Hábitat | Sitios | N sitios | Color |
|---------|--------|----------|-------|
| Bosque nativo | BO-31 a BO-40 | 10 | 🟢 |
| Eucaliptal | EU-16 a EU-20 | 5 | 🟣 |
| Pastizal | PA-16 a PA-20 | 5 | 🟠 |

Período de grabación: **11 de noviembre – 8 de diciembre de 2024**
Grabadora: **AudioMoth** (Open Acoustic Devices) · WAV · 44.100 Hz · mono · 16 bits

---

## Asistencia de IA

El diseño, implementación y depuración de los scripts, el análisis estadístico
y la documentación técnica se realizaron con asistencia de
**Claude Code** (Anthropic) · modelo **claude-sonnet-4-6** ·
modo de razonamiento extendido con presupuesto de esfuerzo máximo
(*high-effort extended reasoning mode*).

Todas las decisiones de diseño experimental, validación ecológica e
interpretación biológica de los resultados fueron realizadas por el autor.

---

## Nota sobre los datos

Los archivos de audio crudos (WAV, ~1 GB/sitio) y los datos de resultados
(CSV, parquet) **no están incluidos en este repositorio** por razones de
confidencialidad y volumen. El repositorio contiene exclusivamente el código
fuente y la documentación.

---

## Documentación técnica

El directorio `06_Documento_Tecnico/` contiene dos documentos LaTeX:

- **`documento_tecnico_paisajes_sonoros.tex`** — Informe formal con metodología
  completa, tablas de resultados, figuras referenciadas y bibliografía. Dirigido
  a investigadores con experiencia en bioacústica.

- **`guia_personal_paisajes_sonoros.tex`** — Guía didáctica en lenguaje cotidiano
  que explica cada paso del pipeline con analogías, versiones de software y
  descripción de cada figura. Incluye glosario completo.

---

## Contacto

**Alcides Ariel Rojas Geraldo**
✉️ alcidesrojasg@gmail.com

Consultas, comentarios y colaboraciones son bienvenidos.
