from flask import Blueprint, render_template

supervised_bp = Blueprint(
    "supervised",
    __name__,
    url_prefix="/ml/supervised"
)

@supervised_bp.route("/")
def supervised_menu():
    return render_template(
        "ml/supervised/menu.html",
        theme="ml"
    )