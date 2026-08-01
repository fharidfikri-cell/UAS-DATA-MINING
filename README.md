# Ujian Akhir Semester (UAS) Data Mining

## Implementasi Supervised dan Unsupervised Learning

### Nama
**Fharid Fikri Syahputra. HS**

### NIM
**23146093**

### Mata Kuliah
Data Mining (SIF304)

### Dosen Pengampu
Teuku Rizky Noviandy, S.Kom., M.Kom.

---

# Deskripsi Proyek

Proyek ini merupakan tugas Ujian Akhir Semester (UAS) Mata Kuliah Data Mining yang mengimplementasikan metode **Supervised Learning** dan **Unsupervised Learning** menggunakan Python dan Streamlit.

Aplikasi ini terdiri dari dua bagian utama:

## 1. Prediksi Risiko Diabetes (Supervised Learning)

Model klasifikasi digunakan untuk memprediksi apakah seorang pasien berisiko mengidap diabetes berdasarkan data kesehatan pasien.

Metode yang digunakan:

- K-Nearest Neighbor (KNN)
- Naïve Bayes
- Decision Tree

Fitur yang digunakan:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

Output aplikasi:

- Prediksi Diabetes / Tidak Diabetes
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 2. Analisis Klaster Lokasi Gerai Kopi (Unsupervised Learning)

Metode K-Means digunakan untuk mengelompokkan lokasi gerai kopi berdasarkan karakteristik wilayah.

Fitur yang digunakan:

- X
- Y
- Population Density
- Traffic Flow
- Competitor Count
- Commercial Area

Output aplikasi:

- Hasil Clustering
- Scatter Plot
- Analisis Zona Ramai / Sedang / Sepi

---

# Dataset

### Dataset Diabetes

Pima Indians Diabetes Dataset

https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### Dataset Gerai Kopi

Dataset Gerai Kopi yang diberikan oleh dosen.

---

# Library yang Digunakan

- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

---

# Cara Menjalankan Aplikasi

### 1. Clone Repository

```bash
git clone https://github.com/USERNAME/UAS-DATA-MINING.git
```

### 2. Masuk ke Folder Project

```bash
cd UAS-DATA-MINING
```

### 3. Install Library

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

```bash
streamlit run app.py
```

---

# Struktur Project

```
UAS_DATAMINING
│
├── app.py
├── train_model.py
├── diabetes.csv
├── lokasi_gerai_kopi_clean.csv
├── requirements.txt
├── README.md
│
└── models
    ├── knn.pkl
    ├── nb.pkl
    ├── dt.pkl
    ├── scaler.pkl
    └── kmeans.pkl
```

---

# Hasil

Aplikasi berhasil:

- Melatih model KNN
- Melatih model Naïve Bayes
- Melatih model Decision Tree
- Melakukan Clustering menggunakan K-Means
- Menampilkan hasil prediksi diabetes
- Menampilkan visualisasi clustering gerai kopi

---

# Link Streamlit

Tambahkan link aplikasi Streamlit setelah deployment.

Contoh:

https://fharidfikri-cell-uas-data-mining-app-be76jj.streamlit.app

---

# Link GitHub

Tambahkan link repository GitHub.

Contoh:

https://github.com/fharidfikri-cell/UAS-DATA-MINING

---

# Lisensi

Project ini dibuat untuk memenuhi tugas **Ujian Akhir Semester (UAS)** Mata Kuliah **Data Mining (SIF304)**.
