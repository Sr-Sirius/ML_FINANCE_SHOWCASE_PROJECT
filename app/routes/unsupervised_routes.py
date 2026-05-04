from flask import Blueprint, render_template
from app.models.unsupervised.kmeans_model import clustering_pipeline

unsupervised_bp = Blueprint(
    "unsupervised",
    __name__,
    url_prefix="/ml/unsupervised"
)

# =========================
# MENU
# =========================
@unsupervised_bp.route("/")
@unsupervised_bp.route("/menu")
def unsupervised_menu():
    return render_template(
        "ml/unsupervised/menu.html",
        theme="ml"
    )

# =========================
# CONCEPTS
# =========================
@unsupervised_bp.route("/concepts")
def unsupervised_concepts():
    return render_template(
        "ml/unsupervised/concepts.html",
        theme="unsupervised"
    )

# =========================
# MANUAL EXERCISE
# =========================
@unsupervised_bp.route("/manual")
def unsupervised_manual():
    return render_template(
        "ml/unsupervised/manual.html",
        theme="unsupervised"
    )

# =========================
# CLUSTERING APPLICATION
# =========================
@unsupervised_bp.route("/application")
def unsupervised_application():

    try:
        result = clustering_pipeline()

        return render_template(
            "ml/unsupervised/application.html",
            **result,
            theme="unsupervised"
        )

    except Exception as e:
        return render_template(
            "ml/unsupervised/application.html",
            data=[],
            centers=[],
            summary={},
            plot=None,
            error=str(e),
            theme="unsupervised"
        )