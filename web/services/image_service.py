import os
from datetime import datetime
from models.db import get_db_connection
from config import config

def handle_upload(file):
    filename = file.filename
    filepath = os.path.join(config.UPLOAD_FOLDER, filename)
    file.save(filepath)

    conn = get_db_connection()
    conn.execute('INSERT INTO images (filename, upload_time) VALUES (?, ?)',
                 (filename, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_all_images():
    conn = get_db_connection()
    images = conn.execute('SELECT * FROM images').fetchall()
    conn.close()
    return images
