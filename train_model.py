import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

# ==========================
# Membuat folder models
# ==========================
os.makedirs("models", exist_ok=True)

# ==========================
# DATASET DIABETES
# ==========================

print("=" * 50)
print("TRAINING MODEL DIABETES")
print("=" * 50)

data = pd.read_csv("diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

for nama, model in models.items():

    model.fit(X_train, y_train)

    prediksi = model.predict(X_test)

    print("\n", nama)
    print("-" * 30)

    print("Accuracy :", round(accuracy_score(y_test, prediksi),4))
    print("Precision:", round(precision_score(y_test, prediksi),4))
    print("Recall   :", round(recall_score(y_test, prediksi),4))
    print("F1 Score :", round(f1_score(y_test, prediksi),4))

    ConfusionMatrixDisplay.from_predictions(y_test, prediksi)
    plt.title(nama)
    plt.show()

    if nama == "KNN":
        joblib.dump(model, "models/knn.pkl")

    elif nama == "Naive Bayes":
        joblib.dump(model, "models/nb.pkl")

    elif nama == "Decision Tree":
        joblib.dump(model, "models/dt.pkl")

print("\nModel Diabetes Berhasil Disimpan")

# ==========================
# DATASET GERAI KOPI
# ==========================

print("\n")
print("=" * 50)
print("TRAINING MODEL K-MEANS")
print("=" * 50)

coffee = pd.read_csv("lokasi_gerai_kopi_clean.csv")

fitur = [
    "x",
    "y",
    "population_density",
    "traffic_flow",
    "competitor_count",
    "is_commercial"
]

X = coffee[fitur]

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

coffee["Cluster"] = kmeans.fit_predict(X)

joblib.dump(kmeans, "models/kmeans.pkl")

print("Model KMeans Berhasil Disimpan")

plt.figure(figsize=(8,6))

plt.scatter(
    coffee["x"],
    coffee["y"],
    c=coffee["Cluster"]
)

plt.title("Clustering Lokasi Gerai Kopi")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()

print("\nSemua model berhasil dibuat.")