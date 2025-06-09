from models.db import get_db_connection

def get_image_by_id(image_id):
    conn = get_db_connection()
    image = conn.execute('SELECT * FROM images WHERE id = ?', (image_id,)).fetchone()
    conn.close()
    return image

def save_annotation(image_id, x, y, w, h, plate_text):
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO annotations (image_id, x, y, width, height, plate_text)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (image_id, x, y, w, h, plate_text))
    conn.commit()
    conn.close()
