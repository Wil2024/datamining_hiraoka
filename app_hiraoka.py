import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import zipfile
import tempfile

# =========================
# CONFIGURACIÓN
# =========================
st.set_page_config(page_title="Simulador Minería de Datos", layout="wide")
st.title("🧠 Simulador de Minería de Datos - Hiraoka E-commerce")
st.markdown("### Aplicando Data Mining: Clustering, Asociación y Clasificación")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("📁 Cargar Archivos")

uploaded_transacciones = st.sidebar.file_uploader(
    "Sube transacciones (Excel)", type=["xlsx"]
)
uploaded_reseñas = st.sidebar.file_uploader(
    "Sube reseñas (Excel)", type=["xlsx"]
)

# =========================
# FUNCIONES AUXILIARES
# =========================
def load_data(transacciones_file, reseñas_file):
    try:
        df_t = pd.read_excel(transacciones_file)
        df_r = pd.read_excel(reseñas_file)
        return df_t, df_r
    except Exception as e:
        st.error(f"Error al cargar archivos: {e}")
        return None, None

def download_excel(df, filename):
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer

# =========================
# MAIN
# =========================
if uploaded_transacciones and uploaded_reseñas:

    df_transacciones, df_reseñas = load_data(
        uploaded_transacciones, uploaded_reseñas
    )

    if df_transacciones is None or df_reseñas is None:
        st.stop()

    # =========================
    # VALIDACIÓN
    # =========================
    required_trans_cols = [
        "customer_id", "order_id", "product_name",
        "category", "quantity", "total_amount"
    ]

    required_review_cols = ["review_text", "sentiment"]

    for col in required_trans_cols:
        if col not in df_transacciones.columns:
            st.error(f"❌ Falta columna en transacciones: {col}")
            st.stop()

    for col in required_review_cols:
        if col not in df_reseñas.columns:
            st.error(f"❌ Falta columna en reseñas: {col}")
            st.stop()

    # =========================
    # VISTA PREVIA
    # =========================
    st.subheader("📊 Vista previa")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Transacciones")
        st.dataframe(df_transacciones.head())

    with col2:
        st.write("Reseñas")
        st.dataframe(df_reseñas.head())

    # =========================
    # 1. CLUSTERING
    # =========================
    st.header("👥 Segmentación de Clientes")

    if st.checkbox("Ejecutar Clustering"):

        customer_features = df_transacciones.groupby("customer_id").agg({
            "total_amount": ["sum", "mean"],
            "order_id": "count",
            "category": lambda x: len(set(x))
        }).reset_index()

        customer_features.columns = [
            "customer_id", "total_spent",
            "avg_spent", "order_count", "unique_categories"
        ]

        X = customer_features[
            ["total_spent", "avg_spent", "order_count", "unique_categories"]
        ]

        X = (X - X.mean()) / X.std()

        k = st.slider("Número de clusters", 2, 6, 3)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        customer_features["cluster"] = kmeans.fit_predict(X)

        # Visualización
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=customer_features,
            x="total_spent",
            y="order_count",
            hue="cluster",
            palette="deep",
            ax=ax
        )
        st.pyplot(fig)

        # Resumen
        st.subheader("📊 Perfil de segmentos")
        summary = customer_features.groupby("cluster").mean(numeric_only=True)
        st.dataframe(summary)

        # Descarga
        st.download_button(
            "📥 Descargar segmentación",
            data=download_excel(customer_features, "segmentacion.xlsx"),
            file_name="segmentacion_clientes.xlsx"
        )

    # =========================
    # 2. CLASIFICACIÓN
    # =========================
    st.header("📝 Análisis de Sentimiento")

    if st.checkbox("Ejecutar Clasificación"):

        X_text = df_reseñas["review_text"]
        y = df_reseñas["sentiment"]

        vectorizer = TfidfVectorizer(max_features=1000)
        X_vec = vectorizer.fit_transform(X_text)

        X_train, X_test, y_train, y_test = train_test_split(
            X_vec, y, test_size=0.2, random_state=42
        )

        model = MultinomialNB()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.text("Reporte:")
        st.text(classification_report(y_test, y_pred))

        # Simulador
        st.subheader("🧪 Prueba en vivo")
        new_text = st.text_input("Escribe una reseña")

        if new_text:
            pred = model.predict(vectorizer.transform([new_text]))[0]
            st.success(f"Predicción: {pred}")

        df_reseñas["predicted"] = model.predict(X_vec)

        st.download_button(
            "📥 Descargar clasificación",
            data=download_excel(df_reseñas, "reseñas.xlsx"),
            file_name="reseñas_clasificadas.xlsx"
        )

    # =========================
    # 3. ASOCIACIÓN
    # =========================
    st.header("🛒 Reglas de Asociación")

    if st.checkbox("Ejecutar Asociación"):

        basket = (
            df_transacciones
            .groupby(["order_id", "product_name"])["quantity"]
            .sum()
            .unstack()
            .fillna(0)
        )

        basket = basket.applymap(lambda x: 1 if x > 0 else 0)

        st.dataframe(basket.head())

        support = st.slider("Soporte mínimo", 0.001, 0.05, 0.01)

        freq = apriori(basket, min_support=support, use_colnames=True)

        if not freq.empty:
            rules = association_rules(freq, metric="lift", min_threshold=1)

            rules["antecedents"] = rules["antecedents"].apply(
                lambda x: ", ".join(list(x))
            )
            rules["consequents"] = rules["consequents"].apply(
                lambda x: ", ".join(list(x))
            )

            st.dataframe(rules.sort_values("confidence", ascending=False))

            st.download_button(
                "📥 Descargar reglas",
                data=download_excel(rules, "reglas.xlsx"),
                file_name="reglas_asociacion.xlsx"
            )
        else:
            st.warning("No se encontraron patrones.")

    # =========================
    # 4. DESCARGA TOTAL
    # =========================
    st.header("📦 Exportar todo")

    if st.button("Descargar ZIP"):

        files = {}

        if 'customer_features' in locals():
            files["segmentacion.xlsx"] = customer_features

        if 'df_reseñas' in locals():
            files["reseñas.xlsx"] = df_reseñas

        if 'rules' in locals():
            files["reglas.xlsx"] = rules

        if len(files) == 0:
            st.warning("No hay resultados para exportar.")
        else:
            temp_dir = tempfile.mkdtemp()
            zip_path = f"{temp_dir}/resultados.zip"

            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for name, df in files.items():
                    path = f"{temp_dir}/{name}"
                    df.to_excel(path, index=False)
                    zipf.write(path, arcname=name)

            with open(zip_path, "rb") as f:
                st.download_button(
                    "📥 Descargar ZIP",
                    data=f,
                    file_name="resultados_datamining.zip"
                )

else:
    st.info("📌 Sube ambos archivos para comenzar.")