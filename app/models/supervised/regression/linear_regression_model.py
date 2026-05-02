import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os

from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use('Agg')


# =========================
# LOAD DATA (MISMA LÓGICA QUE LOGISTIC)
# =========================
def load_data():
    current_dir = os.path.dirname(__file__)

    # Subir hasta raíz del proyecto
    project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))

    file_path = os.path.join(project_root, "data", "finance_data.csv")

    data = pd.read_csv(file_path, delimiter=";")
    data.columns = data.columns.str.strip()

    X = data[["Income", "Previous_Expenses", "Transactions"]]
    y = data["Expense"]

    return X, y, data


# =========================
# TRAIN MODEL
# =========================
def train_model():
    X, y, data = load_data()

    model = LinearRegression()
    model.fit(X, y)

    return model, X, y, data


# =========================
# PREDICTION
# =========================
def predict_expense(income, previous_expenses, transactions):
    model, _, _, _ = train_model()

    input_data = [[income, previous_expenses, transactions]]
    prediction = model.predict(input_data)

    return round(prediction[0], 2)


# =========================
# PLOT
# =========================
def generate_plot():
    model, X, y, data = train_model()

    plt.figure()

    # Scatter real
    plt.scatter(data["Income"], data["Expense"], alpha=0.6)

    # Línea de regresión
    predictions = model.predict(X)
    plt.plot(data["Income"], predictions)

    plt.xlabel("Income")
    plt.ylabel("Expense")
    plt.title("Income vs Expense (Linear Regression)")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()

    plt.close()

    return plot_url