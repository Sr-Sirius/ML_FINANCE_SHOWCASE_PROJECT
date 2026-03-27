from flask import Flask

def create_app():
    app = Flask(__name__)

    from app.routes.main_routes import main
    from app.routes.use_cases_routes import use_cases
    from app.routes.regression_routes import regression_bp

    app.register_blueprint(main)
    app.register_blueprint(use_cases, url_prefix="/use-cases")
    app.register_blueprint(regression_bp, url_prefix="/regression")

    return app