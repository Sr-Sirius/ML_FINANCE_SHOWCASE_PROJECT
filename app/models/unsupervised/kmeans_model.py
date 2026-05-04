import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')


# =========================
# LOAD DATA (MISMA LÓGICA QUE LOS DEMÁS)
# =========================
def load_data():
    current_dir = os.path.dirname(__file__)

    # Subir a raíz del proyecto
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))

    file_path = os.path.join(project_root, "data", "gym_data.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    return pd.read_csv(file_path)


# =========================
# TRAIN MODEL
# =========================
def train_model():
    df = load_data()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(scaled)

    # volver a escala original
    centers = scaler.inverse_transform(model.cluster_centers_)

    return df, centers, model, scaled


# =========================
# PLOT
# =========================
def generate_plot(df, centers):
    plt.figure()

    colors = plt.cm.get_cmap('tab10', len(centers))

    for c in range(len(centers)):
        subset = df[df["cluster"] == c]
        plt.scatter(
            subset["hours"],
            subset["kcal"],
            color=colors(c),
            label=f"Cluster {c}",
            alpha=0.6
        )

    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        color="red",
        marker="X",
        s=200,
        label="Centroids"
    )

    plt.xlabel("Training Hours")
    plt.ylabel("Calories Burned")
    plt.title("K-Means Clustering")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    img = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return img


# =========================
# PIPELINE (FLASK)
# =========================
def clustering_pipeline():
    df, centers, model, scaled = train_model()

    summary = df["cluster"].value_counts().to_dict()
    plot = generate_plot(df, centers)

    return {
        "data": df.head(50).to_dict(orient="records"),
        "centers": centers.tolist(),
        "summary": summary,
        "plot": plot
    }

def generate_elbow(scaled):
    sse_values = []
    k_range = range(1, 10)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled)
        sse_values.append(kmeans.inertia_)

    plt.figure()
    plt.plot(k_range, sse_values, marker='o')

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("SSE")
    plt.title("Elbow Method")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    img = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return img

def clustering_pipeline(k=3):
    df = load_data()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(scaled)

    centers = scaler.inverse_transform(model.cluster_centers_)

    summary = df["cluster"].value_counts().to_dict()
    plot = generate_plot(df, centers)

    sse = model.inertia_
    elbow_plot = generate_elbow(scaled)

    return {
        "data": df.head(50).to_dict(orient="records"),
        "centers": centers.tolist(),
        "summary": summary,
        "plot": plot,
        "sse": sse,
        "elbow_plot": elbow_plot
    }