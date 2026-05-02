import pandas as pd
import os

from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import io
import base64


# =========================
# PATH CONFIG (ROBUSTO)
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
DATA_PATH = os.path.join(BASE_DIR, "app", "data")


# =========================
# LOAD DATA
# =========================
def load_data():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    file_path = os.path.join(project_root, "data", "driver_behavior.csv")

    data = pd.read_csv(file_path)

    X = data[['overspeeding', 'night', 'phone', 'braking']]
    y = data['risk']

    return train_test_split(X, y, test_size=0.3, random_state=42)


# =========================
# TRAIN MODEL
# =========================
def train_model():
    X_train, X_test, y_train, y_test = load_data()

    model = BernoulliNB()
    model.fit(X_train, y_train)

    return model, X_test, y_test


# =========================
# PREDICT
# =========================
def predict_risk(overspeeding, night, phone, braking):
    model, _, _ = train_model()

    input_data = [[overspeeding, night, phone, braking]]

    proba = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    label = "High Risk" if prediction == 1 else "Low Risk"

    return label, round(proba, 4)


# =========================
# EVALUATION
# =========================
def evaluate_model():
    model, X_test, y_test = train_model()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return accuracy, precision, recall, f1, y_test, y_pred, y_prob


# =========================
# CONFUSION MATRIX
# =========================
def plot_confusion_matrix(y_test, y_pred):
    import seaborn as sns

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Low Risk", "High Risk"],
        yticklabels=["Low Risk", "High Risk"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Naive Bayes")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()

    return base64.b64encode(buf.getvalue()).decode()


# =========================
# ROC CURVE
# =========================
def plot_roc(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Naive Bayes")

    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()

    return base64.b64encode(buf.getvalue()).decode()