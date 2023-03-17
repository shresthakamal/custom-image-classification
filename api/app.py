from flask import Flask
from flask_cors import CORS

from api.routes import app_routes


def init_app():
    app = Flask(__name__)

    app_routes(app)

    CORS(app)

    return app


if __name__ == "__main__":
    app = init_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
