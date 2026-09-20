# Antecedentes Literatura Científica
## Proyecto Paisajes Sonoros Tapytá — Campaña Nov-Dic 2024

Tres papers open access seleccionados por su relevancia directa al pipeline
MFCC + PCA + UMAP + HDBSCAN y a los hallazgos de la campaña.

---

## Paper 1 — Metodología (el más parecido al pipeline)

**Archivo:** `Thomas_et_al_2022_Unsupervised_UMAP_HDBSCAN_Vocalizaciones_JAE.pdf`

**Cita completa:**
Thomas, M., Jensen, F.H., Averly, B., Demartsev, V., Manser, M.B., Sainburg, T.,
Roch, M.A., & Strandburg-Peshkin, A. (2022). A practical guide for generating
unsupervised, spectrogram-based latent space representations of animal vocalizations.
*Journal of Animal Ecology*, 91(8), 1567–1581.
DOI: https://doi.org/10.1111/1365-2656.13754

**Por qué leerlo:**
Es prácticamente un manual del mismo pipeline que usamos: UMAP + HDBSCAN aplicado
a representaciones espectrales de vocalizaciones animales, completamente no supervisado
y sin etiquetas manuales. Demuestra que el espacio latente recupera tipos de llamados
biológicamente significativos, respaldando directamente la interpretación de los 3
clusters de anfibio encontrados en PA-17.

**Tamaño:** 4.7 MB (incluye figuras de código y ejemplos)

---

## Paper 2 — Nicho acústico en anuros (contrapunto al hallazgo principal)

**Archivo:** `Villanueva-Rivera_2014_Acoustic_Niche_Partitioning_Anurans_PeerJ.pdf`

**Cita completa:**
Villanueva-Rivera, L.J. (2014). Eleutherodactylus frogs show frequency but no
temporal partitioning: implications for the acoustic niche hypothesis.
*PeerJ*, 2, e496.
DOI: https://doi.org/10.7717/peerj.496

**Por qué leerlo:**
Estudia exactamente la hipótesis del nicho acústico en comunidades de ranas y
encuentra el caso INVERSO al nuestro: las especies particionan la frecuencia pero
NO el tiempo. Nuestro resultado (3 clusters con frecuencia dominante idéntica
~2.850 Hz pero segregación temporal escalonada a 18:00, 21:00 y 00:00 h) es
el complemento opuesto, lo que fortalece la originalidad del hallazgo.
Útil para enmarcar los resultados en el contexto de la literatura.

**Tamaño:** 577 KB

---

## Paper 3 — PAM en el Bosque Atlántico (contexto ecológico)

**Archivo:** `Donnelly_et_al_2026_Acoustic_Niche_Atlantic_Forest_Conservation.pdf`

**Cita completa:**
Donnelly, A., Schork, I., Kaizer, M.C., & Passos, L.F. (2026). Acoustic niche
partitioning and overlap in an anuran community of a threatened Brazilian
Atlantic Forest remnant at Caparaó National Park.
*Conservation*, 6(1), 24.
DOI: https://doi.org/10.3390/conservation6010024

**Por qué leerlo:**
Comunidad de anuros en un remanente del Bosque Atlántico brasileño (hábitat
idéntico al de Tapytá), analizando partición espectral y temporal del espacio
acústico. Publicado en 2026, es el paper más reciente y contextualiza
directamente nuestros resultados dentro del Bosque Atlántico del Alto Paraná.
También usa PCA para cuantificar varianza acústica, igual que nuestro pipeline.

**Tamaño:** 1.4 MB

---

## Resumen de acceso

| Paper | Revista | Acceso | DOI |
|-------|---------|--------|-----|
| Thomas et al. 2022 | J. Animal Ecology (BES/Wiley) | Open Access CC-BY | 10.1111/1365-2656.13754 |
| Villanueva-Rivera 2014 | PeerJ | Open Access CC-BY | 10.7717/peerj.496 |
| Donnelly et al. 2026 | Conservation (MDPI) | Open Access CC-BY | 10.3390/conservation6010024 |
