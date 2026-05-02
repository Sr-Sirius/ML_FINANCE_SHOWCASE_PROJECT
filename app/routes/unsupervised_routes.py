from flask import Blueprint, render_template
from app.models.unsupervised.kmeans_model import clustering_pipeline

unsupervised_bp = Blueprint(
    "unsupervised",
    __name__,
    url_prefix="/unsupervised"
)

# =========================
# MENU PRINCIPAL
# =========================
@unsupervised_bp.route("/")
def unsupervised_menu():
    return render_template(
        "ml/unsupervised/index.html",
        theme="ml"
    )
# =========================
# BASIC CONCEPTS
# =========================
@unsupervised_bp.route("/concepts")
def concepts():
    return render_template(
        "ml/unsupervised/concepts.html",
        theme="ml"
    )

# =========================
# MANUAL EXERCISE
# =========================
@unsupervised_bp.route("/manual")
def manual_kmeans():
    return render_template(
        "ml/unsupervised/manual.html",
        theme="ml"
    )

# =========================
# CLUSTERING APPLICATION
# =========================

@unsupervised_bp.route("/application")
def clustering_app():

    result = clustering_pipeline()

    return render_template(
        "ml/unsupervised/application.html",
        data=result["data"],
        centers=result["centers"],
        summary=result["summary"],
        plot=result["plot"],
        theme="ml"
    )