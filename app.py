import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# ===============================
# LOAD MODEL
# ===============================

knn = joblib.load("models/knn.pkl")
nb = joblib.load("models/nb.pkl")
dt = joblib.load("models/dt.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(
    page_title="UAS Data Mining",
    layout="wide"
)

st.title("UAS DATA MINING")
st.write("Implementasi Supervised dan Unsupervised Learning")

menu = st.sidebar.selectbox(
    "Pilih Menu",
    [
        "Prediksi Diabetes",
        "Clustering Gerai Kopi"
    ]
)

# ======================================================
# PREDIKSI DIABETES
# ======================================================

if menu == "Prediksi Diabetes":

    st.header("Prediksi Risiko Diabetes")

    model = st.selectbox(
        "Pilih Model",
        [
            "KNN",
            "Naive Bayes",
            "Decision Tree"
        ]
    )

    c1, c2 = st.columns(2)

    with c1:

        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 300, 120)
        blood = st.number_input("Blood Pressure", 0, 200, 70)
        skin = st.number_input("Skin Thickness", 0, 100, 20)

    with c2:

        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 1, 120, 25)

    if st.button("Prediksi"):

        data = [[
            pregnancies,
            glucose,
            blood,
            skin,
            insulin,
            bmi,
            dpf,
            age
        ]]

        data = scaler.transform(data)

        if model == "KNN":
            hasil = knn.predict(data)

        elif model == "Naive Bayes":
            hasil = nb.predict(data)

        else:
            hasil = dt.predict(data)

        st.subheader("Hasil Prediksi")

        if hasil[0] == 1:
            st.error("Pasien Diprediksi Mengidap Diabetes")
        else:
            st.success("Pasien Tidak Mengidap Diabetes")

# ======================================================
# CLUSTERING
# ======================================================

else:

    st.header("Analisis Klaster Gerai Kopi")

    coffee = pd.read_csv("lokasi_gerai_kopi_clean.csv")

    fitur = [
        "x",
        "y",
        "population_density",
        "traffic_flow",
        "competitor_count",
        "is_commercial"
    ]

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    coffee["Cluster"] = model.fit_predict(coffee[fitur])

    col1, col2 = st.columns(2)

    with col1:

        x = st.number_input("X", value=50.0)
        y = st.number_input("Y", value=50.0)
        population = st.number_input("Population Density", value=100)
        traffic = st.number_input("Traffic Flow", value=50)
        competitor = st.number_input("Competitor Count", value=2)
        commercial = st.selectbox(
            "Commercial Area",
            [0, 1]
        )

    with col2:

        fig, ax = plt.subplots(figsize=(7,5))

        scatter = ax.scatter(
            coffee["x"],
            coffee["y"],
            c=coffee["Cluster"]
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Clustering Gerai Kopi")

        st.pyplot(fig)

    if st.button("Analisis Lokasi"):

        data_baru = [[
            x,
            y,
            population,
            traffic,
            competitor,
            commercial
        ]]

        cluster = model.predict(data_baru)[0]

        st.subheader("Hasil Analisis")

        st.write("Cluster :", cluster)

        if cluster == 0:
            st.success("Zona Ramai")

        elif cluster == 1:
            st.warning("Zona Sedang")

        else:
            st.error("Zona Sepi")