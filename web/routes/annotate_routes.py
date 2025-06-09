from flask import Blueprint, render_template, request, redirect, url_for, session
from services.annotate_service import get_image_by_id, save_annotation

annotate_bp = Blueprint('annotate', __name__)

@annotate_bp.route('/annotate/<int:image_id>', methods=['GET', 'POST'])
def annotate(image_id):
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    image = get_image_by_id(image_id)
    if request.method == 'POST':
        x = int(request.form['x'])
        y = int(request.form['y'])
        w = int(request.form['w'])
        h = int(request.form['h'])
        plate_text = request.form['plate_text']
        save_annotation(image_id, x, y, w, h, plate_text)
        return redirect(url_for('image.index'))
    return render_template('annotate.html', image=image)