import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression


# Load dataset
data = pd.read_csv("data/finance_data.csv", delimiter=";")
data.columns = data.columns.str.strip()

# Define inputs (X) and output (y)
X = data[["Income", "Previous_Expenses", "Transactions"]]
y = data["Expense"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)


# Prediction function
def predict_expense(income, previous_expenses, transactions):
    input_data = [[income, previous_expenses, transactions]]
    prediction = model.predict(input_data)
    return round(prediction[0], 2)

# Generate plot function and Image encoding for web display
def generate_plot():
    plt.figure()

    # Scatter (real data)
    plt.scatter(data["Income"], data["Expense"])

    # Regression line
    predictions = model.predict(X)
    plt.plot(data["Income"], predictions)

    plt.xlabel("Income")
    plt.ylabel("Expense")
    plt.title("Income vs Expense (Linear Regression)")

    # Save image to memory
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()

    plt.close()

    return plot_url