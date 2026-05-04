from flask import Blueprint, render_template
from app.models.unsupervised.kmeans_model import clustering_pipeline
from flask import request

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
from flask import request

@unsupervised_bp.route("/application", methods=["GET", "POST"])
def unsupervised_application():

    try:
        if request.method == "POST":
            k = int(request.form.get("k", 3))
        else:
            k = 3

        result = clustering_pipeline(k)

        return render_template(
            "ml/unsupervised/application.html",
            **result,
            k_selected=k,
            theme="unsupervised"
        )

    except Exception as e:
        return render_template(
            "ml/unsupervised/application.html",
            data=[],
            centers=[],
            summary={},
            plot=None,
            sse=None,
            elbow_plot=None,
            k_selected=3,
            error=str(e),
            theme="unsupervised"
        )