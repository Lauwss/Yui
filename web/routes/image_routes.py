from flask import Blueprint, render_template, request, redirect, url_for, session
from services.image_service import handle_upload, get_all_images

image_bp = Blueprint('image', __name__)

@image_bp.route('/')
def home():
    return render_template('home.html')

@image_bp.route('/dashboard')
def index():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    images = get_all_images()
    return render_template('index.html', images=images)

@image_bp.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    if request.method == 'POST':
        file = request.files['image']
        handle_upload(file)
        return redirect(url_for('image.index'))
    return render_template('upload.html')