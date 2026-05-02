from flask import Blueprint, render_template, request
from app.models.supervised.regression.linear_regression_model import predict_expense, generate_plot

regression_bp = Blueprint(
    'regression',
    __name__,
    url_prefix="/ml/supervised/regression"
)

# =========================
# CONCEPTS
# =========================
@regression_bp.route('/')
@regression_bp.route('/concepts')
def regression_concepts():
    plot = generate_plot()
    return render_template(
        'ml/supervised/regression/concepts.html',
        plot=plot,
        theme="finance"
    )

# =========================
# APPLICATION
# =========================
@regression_bp.route('/application', methods=['GET', 'POST'])
def regression_application():

    prediction = None
    warning = None
    plot = generate_plot()

    if request.method == 'POST':
        try:
            income = float(request.form['income'])
            previous = float(request.form['previous_expenses'])
            transactions = float(request.form['transactions'])

            if previous > income:
                warning = "Warning: Previous expenses are higher than income. Prediction may be inaccurate."

            prediction = predict_expense(income, previous, transactions)

        except:
            prediction = "Invalid input"
            warning = None

    return render_template(
        'ml/supervised/regression/application.html',
        prediction=prediction,
        warning=warning,
        plot=plot,
        theme="finance"
    )