import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/gym_data.csv"


# =========================
# LOAD DATA
# =========================
def load_data():
    if not os.path.exists(DATA_PATH):
        raise Exception("Dataset not found. Please generate gym_data.csv first.")

    return pd.read_csv(DATA_PATH)


# =========================
# TRAIN MODEL
# =========================
def train_kmeans():
    df = load_data()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(scaled)

    centers = scaler.inverse_transform(model.cluster_centers_)

    return df, centers


# =========================
# PLOT
# =========================
def plot_clusters(df, centers):
    plt.figure()

    colors = ["blue", "green", "orange"]

    for c in range(3):
        subset = df[df["cluster"] == c]
        plt.scatter(
            subset["hours"],
            subset["kcal"],
            color=colors[c],
            label=f"Cluster {c}",
            alpha=0.6
        )

    # centroides
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
    df, centers = train_kmeans()

    summary = df["cluster"].value_counts().to_dict()
    plot = plot_clusters(df, centers)

    return {
        "data": df.head(50).to_dict(orient="records"),
        "centers": centers.tolist(),
        "summary": summary,
        "plot": plot
    }