# app_fase2.py
# ================================================================
# Fase 2 interactiva: Reducción de dimensiones + K-means (Streamlit)
# Autor: Alcides Rojas (adaptado)
# Descripción: Carga .parquet de Fase 1, selecciona features informativos
#              (mfcc_mean_1..3 con posibilidad de sd_1..3), prioriza segmentos
#              con sd alto, aplica reducción (PCA/UMAP), realiza k-means y
#              visualiza clusters con Plotly Express. Incluye filtros por sitio
#              para destacar PA y gestionar BO.
# ================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap

# -----------------------------
# Configuración inicial de la app
# -----------------------------
st.set_page_config(page_title="Fase 2: Dimensionalidad + K-means", layout="wide")

# --- Rutas ---
ruta_salida = Path(r"C:\Users\User\Proyecto_Paisajes_Sonoros_Repositorio_Local\resultados")
ruta_features = ruta_salida / "features.parquet"

# --- Título y descripción ---
st.title("Fase 2: Reducción de dimensiones y K-means sobre MFCCs")
st.write(
    "Visualización interactiva con enfoque acústico: "
    "priorizamos mfcc_mean_1, 2, 3 por su baja redundancia, "
    "segmentos con alta variabilidad (sd) y destacamos sitios como PA (energía útil) "
    "y BO (menos eventos/silencios)."
)

# -----------------------------
# Carga de datos
# -----------------------------
@st.cache_data(show_spinner=True)
def cargar_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df

if not ruta_features.exists():
    st.error(f"No se encontró el archivo: {ruta_features}")
    st.stop()

df = cargar_parquet(ruta_features)

# -----------------------------
# Selección de features
# -----------------------------
st.sidebar.header("Configuración de features")

# Features principales según exploración: mfcc_mean_1..3
features_base = ["mfcc_mean_1", "mfcc_mean_2", "mfcc_mean_3"]

# Opcional: añadir variabilidad para priorizar segmentos dinámicos
add_sd = st.sidebar.checkbox("Incluir desviaciones estándar (sd_1..3)", value=True)
features_sd = ["mfcc_sd_1", "mfcc_sd_2", "mfcc_sd_3"] if add_sd else []
features = [f for f in features_base + features_sd if f in df.columns]

if len(features) < len(features_base):
    st.warning("Faltan columnas mfcc_mean_1..3 en el parquet. Verificá nombres y extracción de Fase 1.")

st.sidebar.write("Features seleccionados:")
st.sidebar.code(features)

# -----------------------------
# Filtros por sitio y sd mínimo
# -----------------------------
st.sidebar.header("Filtros acústicos")
sitios = sorted(df["sitio"].unique()) if "sitio" in df.columns else []
sitios_sel = st.sidebar.multiselect(
    "Seleccionar sitios (recomendación: incluir PA, revisar BO):",
    options=sitios,
    default=[s for s in sitios if "PA" in s] if sitios else []
)

# Filtro por sd mínimo para priorizar segmentos dinámicos
sd_col = "mfcc_sd_1" if "mfcc_sd_1" in df.columns else None
sd_min = st.sidebar.slider(
    "Umbral mínimo de sd_1 (priorizar variabilidad)", min_value=0.0, max_value=100.0, value=10.0, step=1.0
) if sd_col else 0.0

# Posibilidad de excluir BO temporalmente
excluir_bo = st.sidebar.checkbox("Excluir sitios BO (menos eventos/silencios)", value=False)

df_filtrado = df.copy()
if sitios_sel:
    df_filtrado = df_filtrado[df_filtrado["sitio"].isin(sitios_sel)]
if excluir_bo and "sitio" in df_filtrado.columns:
    df_filtrado = df_filtrado[~df_filtrado["sitio"].str.contains("BO", case=False, na=False)]
if sd_col:
    df_filtrado = df_filtrado[df_filtrado[sd_col] >= sd_min]

st.write(f"Segmentos después de filtros: {len(df_filtrado):,}")

if len(df_filtrado) < 50:
    st.warning("Pocos segmentos tras los filtros. Ajustá sitios o sd mínimo para tener suficiente muestra.")
    # No detener; permitir seguir para inspección

# -----------------------------
# Reducción de dimensiones
# -----------------------------
st.sidebar.header("Reducción de dimensiones")
metodo_dim = st.sidebar.selectbox("Método", ["UMAP", "PCA"], index=0)

# Parámetros de UMAP
n_neighbors = st.sidebar.slider("UMAP n_neighbors", 5, 200, 80, 5) if metodo_dim == "UMAP" else None
min_dist = st.sidebar.slider("UMAP min_dist", 0.0, 0.99, 0.10, 0.01) if metodo_dim == "UMAP" else None

# Parámetros de PCA
var_target = st.sidebar.slider("PCA componentes (2–3)", 2, 3, 2, 1) if metodo_dim == "PCA" else 2

# Estándar y proyección
X = df_filtrado[features].dropna()
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

if metodo_dim == "UMAP":
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=123)
    emb = reducer.fit_transform(X_std)  # 2D por defecto
    emb_df = pd.DataFrame({"DIM1": emb[:, 0], "DIM2": emb[:, 1]}, index=X.index)
else:
    pca = PCA(n_components=var_target, random_state=123)
    comps = pca.fit_transform(X_std)
    cols = [f"PC{i+1}" for i in range(comps.shape[1])]
    emb_df = pd.DataFrame(comps[:, :2], columns=cols[:2], index=X.index)
    st.sidebar.write(f"Varianza explicada (PC1+PC2): {pca.explained_variance_ratio_[:2].sum():.2f}")

# Persistir embedding en df_filtrado para hover/descarga
df_filtrado = df_filtrado.loc[X.index].copy()
df_filtrado["DIM1"] = emb_df["DIM1"].values
df_filtrado["DIM2"] = emb_df["DIM2"].values

# -----------------------------
# K-means
# -----------------------------
st.sidebar.header("Clustering K-means")
k = st.sidebar.slider("Número de clusters (k)", 2, 20, 6, 1)

kmeans = KMeans(n_clusters=k, n_init="auto", random_state=123)
labels = kmeans.fit_predict(emb_df.values)
df_filtrado["cluster"] = labels

# -----------------------------
# Visualización principal
# -----------------------------
st.subheader("Mapa de segmentos en espacio reducido")
color_opt = st.selectbox("Color por:", ["cluster", "sitio"] if "sitio" in df_filtrado.columns else ["cluster"])

hover_cols = []
for col in ["archivo_origen", "tiempo_inicio", "tiempo_fin", "mfcc_mean_1", "mfcc_mean_2", "mfcc_mean_3", "mfcc_sd_1"]:
    if col in df_filtrado.columns:
        hover_cols.append(col)

fig = px.scatter(
    df_filtrado,
    x="DIM1", y="DIM2",
    color=color_opt,
    hover_data=hover_cols,
    opacity=0.85,
    height=700,
    template="plotly_white",
)
fig.update_traces(marker=dict(size=6))
fig.update_layout(
    legend_title_text=color_opt.capitalize(),
    margin=dict(l=10, r=10, t=30, b=10),
    title=f"{metodo_dim} + K-means (k={k}) | Filtros: sd_min={sd_min}, sitios={', '.join(sitios_sel) if sitios_sel else 'todos'}"
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Resúmenes por cluster y por sitio
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Resumen por cluster")
    if "sitio" in df_filtrado.columns:
        resumen_cluster = df_filtrado.groupby("cluster").agg(
            conteo=("cluster", "size"),
            sitios_unicos=("sitio", lambda s: ", ".join(sorted(set(s)))),
            mean1=("mfcc_mean_1", "mean"),
            mean2=("mfcc_mean_2", "mean"),
            mean3=("mfcc_mean_3", "mean"),
            sd1=("mfcc_sd_1", "mean") if "mfcc_sd_1" in df_filtrado.columns else ("cluster", "size")
        )
    else:
        resumen_cluster = df_filtrado.groupby("cluster").agg(
            conteo=("cluster", "size"),
            mean1=("mfcc_mean_1", "mean"),
            mean2=("mfcc_mean_2", "mean"),
            mean3=("mfcc_mean_3", "mean"),
            sd1=("mfcc_sd_1", "mean") if "mfcc_sd_1" in df_filtrado.columns else ("cluster", "size")
        )
    st.dataframe(resumen_cluster)

with col2:
    if "sitio" in df_filtrado.columns:
        st.subheader("Resumen por sitio")
        resumen_sitio = df_filtrado.groupby("sitio").agg(
            conteo=("sitio", "size"),
            mediana_mean1=("mfcc_mean_1", "median"),
            mediana_mean2=("mfcc_mean_2", "median"),
            sd1_prom=("mfcc_sd_1", "mean") if "mfcc_sd_1" in df_filtrado.columns else ("sitio", "size")
        )
        st.dataframe(resumen_sitio)

# -----------------------------
# Descarga de resultados
# -----------------------------
st.subheader("Descargar resultados")
out_cols = ["DIM1", "DIM2", "cluster"] + features + ["sitio", "archivo_origen", "tiempo_inicio", "tiempo_fin"]
out_cols = [c for c in out_cols if c in df_filtrado.columns]
csv_bytes = df_filtrado[out_cols].to_csv(index=False).encode("utf-8")
st.download_button("Descargar CSV de embedding + clusters", data=csv_bytes, file_name="fase2_embedding_clusters.csv", mime="text/csv")

st.success("Listo. Ajustá k, método de reducción y filtros para explorar tus clusters acústicos.")