import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ------------------------------------------------------------
# Konfigurasi Halaman
# ------------------------------------------------------------
st.set_page_config(page_title="UAS Data Mining", layout="wide")
st.title("🎯 Ujian Akhir Semester - Data Mining")
st.markdown("**Nama:** FHARID FIKRI SYAHPUTRA.HS | **NIM:** 23146093")
st.divider()

# ------------------------------------------------------------
# Sidebar Navigasi
# ------------------------------------------------------------
st.sidebar.title("📌 Navigasi")
menu = st.sidebar.radio(
    "Pilih Menu:",
    ["🏠 Home", "🩺 Prediksi Diabetes", "📍 Clustering Gerai Kopi"]
)

# ------------------------------------------------------------
# FUNGSI LOAD DATA (dengan caching)
# ------------------------------------------------------------
@st.cache_data
def load_diabetes_data():
    df = pd.read_csv('diabetes.csv')
    return df

@st.cache_data
def load_cafe_data():
    df = pd.read_csv('lokasi_gerai_kopi_clean.csv')
    return df

# ------------------------------------------------------------
# FUNGSI BUILD MODEL DIABETES
# ------------------------------------------------------------
@st.cache_resource
def build_diabetes_models(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5)
    }
    
    results = {}
    for name, model in models.items():
        if name == 'KNN':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'scaler': scaler if name == 'KNN' else None,
            'X_train': X_train if name != 'KNN' else X_train_scaled
        }
    
    return results

# ------------------------------------------------------------
# FUNGSI CLUSTERING K-MEANS
# ------------------------------------------------------------
@st.cache_resource
def perform_clustering(df, n_clusters=3):
    features = ['x', 'y', 'population_density', 'traffic_flow', 
                'competitor_count', 'is_commercial']
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clustered = df.copy()
    df_clustered['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analisis zona sepi (klaster dengan kepadatan terendah)
    cluster_analysis = df_clustered.groupby('Cluster').agg({
        'population_density': 'mean',
        'traffic_flow': 'mean',
        'competitor_count': 'mean'
    }).reset_index()
    
    sepi_cluster = cluster_analysis.loc[
        cluster_analysis['population_density'].idxmin(), 'Cluster'
    ]
    
    df_clustered['Zone'] = df_clustered['Cluster'].apply(
        lambda x: '🟥 Sepi' if x == sepi_cluster else '🟩 Ramai'
    )
    
    return df_clustered, kmeans, scaler, sepi_cluster

# ------------------------------------------------------------
# HOME
# ------------------------------------------------------------
if menu == "🏠 Home":
    st.header("📋 Deskripsi Proyek")
    st.markdown("""
    **Mata Kuliah** : Data Mining (SIF304)  
    **Tahun Ajaran** : Genap 2025/2026  
    **Dosen Pengampu** : Teuku Rizky Noviandy, S.Kom., M.Kom.  

    ---

    ### 🎯 Tujuan Proyek
    Membangun aplikasi berbasis web dengan **Streamlit** yang mengimplementasikan dua model data mining:

    1. **Klasifikasi Diabetes** menggunakan tiga algoritma:
       - K-Nearest Neighbors (KNN)
       - Naive Bayesian
       - Decision Tree

    2. **Clustering Lokasi Gerai Kopi** menggunakan K-Means untuk mengidentifikasi zona sepi.

    ---

    ### 📊 Dataset
    - **Diabetes** : [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
    - **Gerai Kopi** : [Dataset Spasial](https://drive.google.com/file/d/1QuDHd-pebfOY-DoyFACtMyK-_UyuMJP8/view?usp=sharing)

    ---

    ### 🔧 Teknologi yang Digunakan
    - Python, Streamlit, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Folium
    """)

# ------------------------------------------------------------
# BAGIAN A: KLASIFIKASI DIABETES
# ------------------------------------------------------------
elif menu == "🩺 Prediksi Diabetes":
    st.header("🩺 Prediksi Risiko Diabetes Berdasarkan Data Pasien")
    st.markdown("""
    **Deskripsi:** Aplikasi ini menggunakan 3 metode klasifikasi (KNN, Naive Bayes, Decision Tree) 
    untuk memprediksi apakah seorang pasien mengidap diabetes berdasarkan data medis seperti 
    kadar glukosa, BMI, usia, dan lainnya.
    """)
    
    # Load data dan bangun model
    df_diabetes = load_diabetes_data()
    results_diabetes = build_diabetes_models(df_diabetes)
    
    # TAB: Evaluasi Model & Prediksi
    tab1, tab2 = st.tabs(["📊 Evaluasi Model", "🔮 Prediksi Pasien Baru"])
    
    with tab1:
        st.subheader("📊 Perbandingan Metrik Evaluasi Model")
        
        # Tabel metrik
        metrics_df = pd.DataFrame({
            model: {
                'Akurasi': f"{results_diabetes[model]['Accuracy']:.2%}",
                'Precision': f"{results_diabetes[model]['Precision']:.2%}",
                'Recall': f"{results_diabetes[model]['Recall']:.2%}",
                'F1-Score': f"{results_diabetes[model]['F1-Score']:.2%}"
            }
            for model in results_diabetes.keys()
        }).T
        
        st.dataframe(metrics_df, use_container_width=True)
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        metrics_plot = pd.DataFrame({
            'Akurasi': [results_diabetes[m]['Accuracy'] for m in results_diabetes.keys()],
            'Precision': [results_diabetes[m]['Precision'] for m in results_diabetes.keys()],
            'Recall': [results_diabetes[m]['Recall'] for m in results_diabetes.keys()],
            'F1-Score': [results_diabetes[m]['F1-Score'] for m in results_diabetes.keys()]
        }, index=results_diabetes.keys())
        
        metrics_plot.plot(kind='bar', ax=ax, rot=0)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title('Perbandingan Metrik Model')
        ax.legend(loc='lower right')
        st.pyplot(fig)
        
        # Confusion Matrix
        st.subheader("📊 Confusion Matrix")
        col1, col2, col3 = st.columns(3)
        
        for idx, (name, data) in enumerate(results_diabetes.items()):
            cm = confusion_matrix(data['y_test'], data['y_pred'])
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Tidak Diabetes', 'Diabetes'],
                        yticklabels=['Tidak Diabetes', 'Diabetes'])
            ax.set_title(f'{name}')
            ax.set_xlabel('Prediksi')
            ax.set_ylabel('Aktual')
            
            col = [col1, col2, col3][idx]
            with col:
                st.pyplot(fig)
    
    with tab2:
        st.subheader("🔮 Prediksi Diabetes Pasien Baru")
        
        # Pilih model
        selected_model = st.selectbox(
            "Pilih Model Klasifikasi:",
            list(results_diabetes.keys())
        )
        
        # Input fitur
        st.markdown("### Masukkan Data Pasien")
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input('Jumlah Kehamilan (Pregnancies)', 0, 20, 0)
            glucose = st.number_input('Kadar Glukosa (Glucose)', 0, 250, 100)
            blood_pressure = st.number_input('Tekanan Darah (BloodPressure)', 0, 150, 70)
            skin_thickness = st.number_input('Ketebalan Kulit (SkinThickness)', 0, 100, 20)
        
        with col2:
            insulin = st.number_input('Insulin', 0, 900, 80)
            bmi = st.number_input('BMI', 0.0, 70.0, 25.0)
            dpf = st.number_input('Diabetes Pedigree Function', 0.0, 3.0, 0.5)
            age = st.number_input('Usia (Age)', 1, 120, 30)
        
        if st.button("🔍 Prediksi", type="primary"):
            input_data = np.array([[pregnancies, glucose, blood_pressure, 
                                   skin_thickness, insulin, bmi, dpf, age]])
            
            model_data = results_diabetes[selected_model]
            
            if selected_model == 'KNN':
                input_scaled = model_data['scaler'].transform(input_data)
                pred = model_data['model'].predict(input_scaled)
            else:
                pred = model_data['model'].predict(input_data)
            
            # Tampilkan hasil
            st.markdown("### 📋 Hasil Prediksi")
            if pred[0] == 1:
                st.error("🚨 **POSITIF DIABETES**")
                st.warning("Pasien diprediksi mengidap diabetes. Segera lakukan konsultasi medis.")
            else:
                st.success("✅ **NEGATIF DIABETES**")
                st.success("Pasien diprediksi tidak mengidap diabetes.")
            
            # Probabilitas jika tersedia
            if hasattr(model_data['model'], 'predict_proba'):
                proba = model_data['model'].predict_proba(input_data)[0]
                st.info(f"Probabilitas: Tidak Diabetes = {proba[0]:.2%}, Diabetes = {proba[1]:.2%}")

# ------------------------------------------------------------
# BAGIAN B: CLUSTERING GERAI KOPI
# ------------------------------------------------------------
elif menu == "📍 Clustering Gerai Kopi":
    st.header("📍 Analisis Klaster Lokasi Gerai Kopi dan Deteksi Zona Sepi")
    st.markdown("""
    **Deskripsi:** Menggunakan K-Means Clustering untuk mengelompokkan lokasi gerai kopi 
    berdasarkan data spasial dan parameter lingkungan. Tujuannya adalah mengidentifikasi 
    zona dengan potensi pelanggan rendah (zona sepi).
    
    **Fitur yang digunakan:** Koordinat (x,y), kepadatan penduduk, arus lalu lintas, 
    jumlah kompetitor, dan status komersial.
    """)
    
    # Load data dan clustering
    df_cafe = load_cafe_data()
    df_clustered, kmeans_model, scaler_cafe, sepi_cluster = perform_clustering(df_cafe, n_clusters=3)
    
    # TAB: Visualisasi & Prediksi
    tab1, tab2 = st.tabs(["🗺️ Visualisasi Clustering", "📍 Prediksi Lokasi Baru"])
    
    with tab1:
        st.subheader("🗺️ Visualisasi Persebaran Gerai Kopi")
        
        # Statistik klaster
        cluster_stats = df_clustered.groupby('Cluster').agg({
            'population_density': ['mean', 'min', 'max'],
            'traffic_flow': ['mean', 'min', 'max'],
            'competitor_count': ['mean', 'min', 'max']
        }).round(2)
        
        st.dataframe(cluster_stats, use_container_width=True)
        
        # Informasi zona sepi
        sepi_info = df_clustered[df_clustered['Zone'] == '🟥 Sepi']
        st.info(f"📍 **Zona Sepi Terdeteksi:** {len(sepi_info)} gerai berada di cluster {sepi_cluster} "
                f"(kepadatan rata-rata: {sepi_info['population_density'].mean():.0f})")
        
        # Scatter Plot
        st.subheader("📊 Scatter Plot Klaster")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for cluster in sorted(df_clustered['Cluster'].unique()):
            data = df_clustered[df_clustered['Cluster'] == cluster]
            ax.scatter(data['x'], data['y'], label=f"Cluster {cluster}", alpha=0.7, s=50)
        
        # Tandai zona sepi
        sepi_data = df_clustered[df_clustered['Zone'] == '🟥 Sepi']
        ax.scatter(sepi_data['x'], sepi_data['y'], 
                  facecolors='none', edgecolors='red', s=200, 
                  label='Zona Sepi', linewidths=2)
        
        # Centroid
        centroids = kmeans_model.cluster_centers_
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                  c='black', marker='X', s=300, label='Centroid')
        
        ax.set_xlabel('Longitude (x)')
        ax.set_ylabel('Latitude (y)')
        ax.set_title('Persebaran Gerai Kopi Berdasarkan Klaster')
        ax.legend()
        st.pyplot(fig)
        
        # Peta Interaktif Folium
        st.subheader("🗺️ Peta Interaktif")
        
        center_lat = df_clustered['y'].mean()
        center_lon = df_clustered['x'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        for _, row in df_clustered.iterrows():
            color = colors[row['Cluster'] % len(colors)]
            
            if row['Zone'] == '🟥 Sepi':
                folium.Marker(
                    [row['y'], row['x']],
                    popup=f"Cluster {row['Cluster']} - ZONA SEPI",
                    tooltip="⚠️ Zona Sepi",
                    icon=folium.Icon(color='black', icon='exclamation-triangle', prefix='fa')
                ).add_to(m)
            else:
                folium.CircleMarker(
                    [row['y'], row['x']],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.6,
                    popup=f"Cluster {row['Cluster']}"
                ).add_to(m)
        
        for i, center in enumerate(kmeans_model.cluster_centers_):
            folium.Marker(
                [center[1], center[0]],
                icon=folium.Icon(color='darkblue', icon='info-sign'),
                popup=f"Centroid Cluster {i}"
            ).add_to(m)
        
        st_folium(m, width=700, height=500)
    
    with tab2:
        st.subheader("📍 Prediksi Klaster dan Zona Lokasi Baru")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_x = st.number_input('Longitude (x)', value=float(df_clustered['x'].mean()))
            new_y = st.number_input('Latitude (y)', value=float(df_clustered['y'].mean()))
            new_pop = st.number_input('Population Density', value=1000.0)
        
        with col2:
            new_traffic = st.number_input('Traffic Flow', value=500.0)
            new_competitor = st.number_input('Competitor Count', value=2, step=1)
            new_commercial = st.selectbox('Is Commercial', [0, 1])
        
        if st.button("🔍 Prediksi Klaster", type="primary"):
            new_data = np.array([[new_x, new_y, new_pop, new_traffic, 
                                 new_competitor, new_commercial]])
            new_scaled = scaler_cafe.transform(new_data)
            cluster_pred = kmeans_model.predict(new_scaled)[0]
            
            is_sepi = (cluster_pred == sepi_cluster)
            
            st.markdown("### 📋 Hasil Prediksi:")
            st.info(f"📍 Lokasi masuk **Cluster {cluster_pred}**")
            
            if is_sepi:
                st.error("🔴 **ZONA SEPI** - Potensi pelanggan rendah")
                st.warning("Rekomendasi: Pertimbangkan untuk meningkatkan promosi atau menambah layanan.")
            else:
                st.success("🟢 **ZONA RAMAI** - Potensi pelanggan tinggi")
                st.success("Rekomendasi: Lokasi strategis untuk membuka gerai baru.")
            
            # Karakteristik cluster
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_pred]
            st.markdown("### 📊 Karakteristik Cluster:")
            st.dataframe(cluster_data[['population_density', 'traffic_flow', 'competitor_count']].describe())