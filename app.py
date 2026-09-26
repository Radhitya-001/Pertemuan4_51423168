import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Segmentasi Pelanggan Kartu Kredit", layout="centered")

# -----------------------------
# Load model & artefak pendukung
# -----------------------------
MODEL_PATH = "model/kmeans_pipeline.joblib"
PROFILE_PATH = "model/cluster_profile.csv"

@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle["features"], bundle["final_k"]

if not os.path.exists(MODEL_PATH):
    st.error(
        f"File model tidak ditemukan di '{MODEL_PATH}'. "
        "Pastikan folder 'model/' (berisi kmeans_pipeline.joblib) "
        "diletakkan satu folder dengan app.py ini."
    )
    st.stop()

pipeline, feature_cols, final_k = load_model()

# -----------------------------
# Header
# -----------------------------
st.title("Segmentasi Perilaku Pemegang Kartu Kredit")
st.write(
    "Aplikasi ini memprediksi **cluster (segmen)** pelanggan kartu kredit "
    "berdasarkan model K-Means yang telah dilatih menggunakan metodologi CRISP-DM."
)
st.caption(f"Model menggunakan {final_k} cluster dan fitur: {', '.join(feature_cols)}")

st.divider()

# -----------------------------
# Input pengguna
# -----------------------------
st.subheader("Masukkan Data Pelanggan")

input_values = {}
default_hints = {
    "BALANCE": 1500.0,
    "PURCHASES": 750.0,
    "CREDIT_LIMIT": 4500.0,
}

cols = st.columns(len(feature_cols))
for i, feat in enumerate(feature_cols):
    with cols[i]:
        input_values[feat] = st.number_input(
            feat,
            min_value=0.0,
            value=default_hints.get(feat, 0.0),
            step=100.0,
            format="%.2f",
        )

predict_btn = st.button("Prediksi Cluster", type="primary")

# -----------------------------
# Prediksi
# -----------------------------
if predict_btn:
    input_df = pd.DataFrame([input_values], columns=feature_cols)
    cluster_pred = pipeline.predict(input_df)[0]

    st.success(f"Pelanggan ini diprediksi masuk ke **Cluster {cluster_pred}**")

    if os.path.exists(PROFILE_PATH):
        st.subheader("Profil Rata-rata Cluster (dari data training)")
        profile_df = pd.read_csv(PROFILE_PATH, index_col=0)
        st.dataframe(profile_df)

        if cluster_pred in profile_df.columns.astype(str).tolist() or str(cluster_pred) in profile_df.columns:
            st.info(
                f"Bandingkan nilai input di atas dengan kolom Cluster {cluster_pred} "
                "pada tabel untuk melihat karakteristik segmen tersebut."
            )
    else:
        st.warning(
            "File profil cluster (cluster_profile.csv) tidak ditemukan, "
            "jadi hanya nomor cluster yang ditampilkan."
        )

st.divider()
st.caption(
    "Dibuat sebagai bagian dari Tugas Mandiri Pertemuan 4 — "
    "Kursus Data Science, Universitas Gunadarma."
)
