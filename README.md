# Benchmark Bioacústico en Disco Duro Externo

Este módulo evalúa el rendimiento y estructura de datos de un disco duro externo (500 GB) que contiene grabaciones pasivas de ranas en la Reserva Natural Tapytá (Caazapá, Paraguay). Se aplican dos fases de procesamiento:

---

## Fase 1: Extracción de MFCCs

- Segmentación de audios en ventanas de 4 segundos.
- Filtro pasa banda (700–2500 Hz) y reducción de ruido.
- Cálculo de 20 coeficientes MFCC + desviaciones estándar.
- Salida: `features.parquet` con metadatos por segmento.

---

## Fase 2: Reducción y Clustering

- Estandarización de MFCCs.
- Reducción de dimensionalidad con UMAP.
- Agrupamiento con HDBSCAN.
- Exportación de anotaciones en formato Raven (`.Table.1.selections.txt`).
- Visualización de clusters (`clusters.png`).

---

## Objetivo del benchmark

- Evaluar la calidad y organización de los datos del disco externo.
- Detectar grabaciones útiles vs. ruido.
- Identificar patrones acústicos por sitio/punto de muestreo.
- Preparar datos para validación humana y clasificación de especies.

---


## Requisitos

- Python ≥ 3.8  
- pandas, numpy, librosa, soundfile, umap-learn, hdbscan, matplotlib, seaborn, noisereduce

---

## Validación

Las anotaciones Raven generadas permiten inspección visual y auditiva. Se recomienda validar ≥5 segmentos por cluster y propagar etiquetas si la pureza es alta.

---

## Contacto

Desarrollado por **Alcides Rojas**  
📧 alcidesrojasg@gmail.com  
🧠 Asistencia técnica: Microsoft Copilot
