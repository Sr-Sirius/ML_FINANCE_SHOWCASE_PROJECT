from flask import Blueprint, render_template

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
    return render_template(
        "ml/unsupervised/application.html",
        theme="ml"
    )