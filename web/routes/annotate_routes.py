from services.annotate_service import get_image_by_id, save_annotation
from services.full_car_recognition import recognize_plate_and_location
from flask import Blueprint, render_template, request, redirect, url_for, session
import os

annotate_bp = Blueprint('annotate', __name__)

@annotate_bp.route('/annotate/<int:image_id>', methods=['GET', 'POST'])
def annotate(image_id):
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    image = get_image_by_id(image_id)

    # 获取项目根目录，并拼接为本地绝对路径
    project_root = os.path.abspath(os.path.dirname(__file__))
    image_path = os.path.abspath(os.path.join('static', 'uploads', image['filename']))

    plate_text = None
    error = None

    # 判断图片路径是否存在，避免 PaddleOCR 报错
    if not os.path.exists(image_path):
        error = f"图像文件 {image_path} 不存在"
        return render_template('annotate.html', image=image, error=error)

    if request.method == 'POST':
        plate_text, box, error = recognize_plate_and_location(image_path)

        if plate_text and box:
            x = int(min([p[0] for p in box]))
            y = int(min([p[1] for p in box]))
            w = int(max([p[0] for p in box])) - x
            h = int(max([p[1] for p in box])) - y

            save_annotation(image_id, x, y, w, h, plate_text)
            return render_template(
                'annotate.html',
                image=image,
                plate_text=plate_text,
                box={'x': x, 'y': y, 'w': w, 'h': h}
            )
        else:
            return render_template(
                'annotate.html',
                image=image,
                error=error
            )

    return render_template('annotate.html', image=image)
