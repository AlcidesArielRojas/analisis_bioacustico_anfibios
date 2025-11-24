# Análisis No Supervisado de Bioacústica en Anfibios

Este repositorio contiene un pipeline completo para el análisis no supervisado de grabaciones pasivas de ranas en la Reserva Natural Tapytá (Caazapá, Paraguay). Incluye preprocesamiento, extracción de características (MFCCs), reducción de dimensionalidad (UMAP), clustering (HDBSCAN) y exportación en formato Raven para validación humana.

Consultas y recomendaciones: alcidesrojasg@gmail.com

## Estructura del repositorio (A priori)

- `fase1_extraccion.py`: Extracción de MFCCs desde audios `.wav`/`.flac`.
- `fase2_clustering.py`: Reducción UMAP, clustering HDBSCAN y exportación Raven.
- `requirements.txt`: Dependencias del proyecto.
- `README.md`: Documentación del proyecto.


---

### Ejemplo de salida

## Ejemplo de anotación Raven exportada

| Selection | Begin Time (s) | End Time (s) | Cluster_ID | Cluster_Prob | Species_Label |
|-----------|----------------|--------------|------------|---------------|----------------|
| 1         | 0.0            | 4.0          | Cluster 2  | 0.87          | Scinax sp.     |
| 2         | 4.0            | 8.0          | Cluster 2  | 0.91          | Scinax sp.     |
| 3         | 8.0            | 12.0         | Ruido      | 0.00          |                |

> Estas anotaciones pueden abrirse en Raven para inspección visual y auditiva. La columna `Cluster_Prob` indica la confianza del clustering, y `Species_Label` puede completarse manualmente durante la validación humana.

## Validación humana

Las anotaciones se exportan en formato compatible con Raven para inspección visual y auditiva. Se recomienda validar 5–10 anotaciones por cluster y propagar etiquetas si la pureza es ≥ 80%. La columna `cluster_prob` permite priorizar anotaciones confiables.


## Requisitos

- Python ≥ 3.8
- pandas, numpy, librosa, soundfile, umap-learn, hdbscan, matplotlib, seaborn, tqdm, noisereduce

## Créditos

Desarrollado por **Alcides Rojas** con asistencia de Microsoft Copilot.

