from flask import Blueprint, render_template, request

from app.models.supervised.classification.logistic_model import (
    predict_risk,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc,
    train_model
)

from app.models.supervised.classification.bernoulli_nb_model import (
    predict_risk as nb_predict,
    evaluate_model as nb_evaluate,
    plot_confusion_matrix as nb_cm,
    plot_roc as nb_roc,
    train_model as nb_train
)

classification_bp = Blueprint(
    "classification",
    __name__,
    url_prefix="/ml/supervised/classification"
)

# =========================
# Logistic Regression - Concepts
# =========================
@classification_bp.route("/logistic/concepts")
def logistic_concepts():
    return render_template(
        "ml/supervised/classification/logistic/concepts.html",
        theme="cars"
    )

# =========================
# Logistic Regression - Application
# =========================
@classification_bp.route("/logistic/application", methods=["GET", "POST"])
def logistic_application():

    prediction = None
    probability = None

    if request.method == "POST":
        try:
            speed = float(request.form["speed"])
            trips = float(request.form["trips"])
            time = float(request.form["time"])

            prediction, probability = predict_risk(speed, trips, time)

        except:
            prediction = "Invalid input"
            probability = None

    accuracy, precision, recall, f1, y_test, y_pred, y_prob = evaluate_model()

    cm_plot = plot_confusion_matrix(y_test, y_pred)

    roc_plot = plot_roc(y_test, y_prob)

    return render_template(
        "ml/supervised/classification/logistic/application.html",
        prediction=prediction,
        probability=probability,
        accuracy=round(accuracy, 2),
        precision=round(precision, 2),
        recall=round(recall, 2),
        f1=round(f1, 2),
        cm_plot=cm_plot,
        roc_plot=roc_plot,
        theme="cars"
    )

# =========================
# Naive Bayes - Concepts
# =========================
@classification_bp.route("/naive-bayes/concepts")
def nb_concepts():
    return render_template(
        "ml/supervised/classification/naive_bayes/concepts.html",
        theme="cars"
    )

# =========================
# Naive Bayes - Application
# =========================
@classification_bp.route("/naive-bayes/application", methods=["GET", "POST"])
def nb_application():

    prediction = None
    probability = None

    if request.method == "POST":
        try:
            overspeeding = int(request.form["overspeeding"])
            night = int(request.form["night"])
            phone = int(request.form["phone"])
            braking = int(request.form["braking"])

            prediction, probability = nb_predict(
                overspeeding,
                night,
                phone,
                braking
            )
        except:
            prediction = "Invalid input"
            probability = None

    accuracy, precision, recall, f1, y_test, y_pred, y_prob = nb_evaluate()

    cm_plot = nb_cm(y_test, y_pred)
    roc_plot = nb_roc(y_test, y_prob)

    return render_template(
        "ml/supervised/classification/naive_bayes/application.html",
        prediction=prediction,
        probability=probability,
        accuracy=round(accuracy, 2),
        precision=round(precision, 2),
        recall=round(recall, 2),
        f1=round(f1, 2),
        cm_plot=cm_plot,
        roc_plot=roc_plot,
        theme="cars"
    )