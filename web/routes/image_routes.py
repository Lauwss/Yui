from flask import Blueprint, render_template, request, redirect, url_for, session,flash
from services.image_service import handle_upload, get_all_images
from models.db import get_db_connection
import os
import config
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

@image_bp.route('/delete/<int:image_id>', methods=['POST'])
def delete_image(image_id):
    conn = get_db_connection()
    image = conn.execute('SELECT * FROM images WHERE id = ?', (image_id,)).fetchone()
    
    if image:
        filepath = os.path.join(r'D:\car_board_recognize\Yui\web\static\uploads', image['filename'])
        if os.path.exists(filepath):
            os.remove(filepath)

        conn.execute('DELETE FROM images WHERE id = ?', (image_id,))
        conn.commit()
    
    conn.close()
    flash("图片已删除")
    return redirect(url_for('image.index'))