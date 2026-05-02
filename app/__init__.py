from flask import Flask

def create_app():
    app = Flask(__name__)

    from app.routes.main_routes import main_bp
    from app.routes.use_cases_routes import use_cases
    from app.routes.regression_routes import regression_bp
    from app.routes.classification_routes import classification_bp
    from app.routes.unsupervised_routes import unsupervised_bp
    from app.routes.supervised_routes import supervised_bp  

    # =========================
    # REGISTER BLUEPRINTS
    # =========================
    app.register_blueprint(main_bp)

    app.register_blueprint(use_cases, url_prefix="/use-cases")

    app.register_blueprint(supervised_bp)
    app.register_blueprint(regression_bp)
    app.register_blueprint(classification_bp)
    app.register_blueprint(unsupervised_bp)

    return app