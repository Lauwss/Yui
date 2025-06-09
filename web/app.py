## license_plate_app/app.py
from flask import Flask
from models.db import init_db
from routes.image_routes import image_bp
from routes.annotate_routes import annotate_bp
from routes.auth_routes import auth_bp
import os

app = Flask(__name__)
app.config.from_pyfile('config/config.py')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

app.secret_key = app.config['SECRET_KEY']
app.register_blueprint(image_bp)
app.register_blueprint(annotate_bp)
app.register_blueprint(auth_bp)

init_db()

if __name__ == '__main__':
    app.run(debug=True)
