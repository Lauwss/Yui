# services/full_car_recognition.py

def recognize_plate_and_location(image_path):
    from tools.car import load_image, detect_colors, locate_license_plate, preprocess_image
    from tools.car import ocr_primary, simple_ocr, plt_show0
    import cv2
    import numpy as np
    import os

    try:
        image = load_image(image_path)
        image_detect = detect_colors(image)
        exam = locate_license_plate(image_detect)
        if exam is None:
            return None, None, "车牌定位失败"

        cv2.imwrite('exam.png', exam)
        image = preprocess_image(exam)
        output_image = image.copy()

        temp_path = 'exam.png'
        result = ocr_primary.ocr(temp_path, cls=True)

        license_plate = []
        boxes = []
        scores = []

        if result and result[0]:
            for idx, line in enumerate(result[0]):
                box = line[0]
                text = line[1][0]
                score = line[1][1]

                x_min = int(min([p[0] for p in box])) - 5
                x_max = int(max([p[0] for p in box])) + 5
                y_min = int(min([p[1] for p in box])) - 5
                y_max = int(max([p[1] for p in box])) + 5

                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(image.shape[1], x_max)
                y_max = min(image.shape[0], y_max)

                text_img = image[y_min:y_max, x_min:x_max]
                text_path = f"text_{idx+1}.jpg"
                cv2.imwrite(text_path, text_img)

                re_text, re_score = simple_ocr(text_path)
                if re_text:
                    license_plate.append(re_text)
                    boxes.append(box)
                    scores.append(re_score)
                else:
                    license_plate.append(text)
                    boxes.append(box)
                    scores.append(score)

            # 选出一个车牌
            plate_text = ""
            if license_plate:
                candidates = [(t, s, b) for t, s, b in zip(license_plate, scores, boxes) if 6 <= len(t.strip()) <= 9]
                if candidates:
                    best_text, best_score, best_box = max(candidates, key=lambda x: x[1])
                    plate_text = best_text
                    box = best_box
                else:
                    plate_text = license_plate[0]
                    box = boxes[0]

                return plate_text, box, None

            else:
                return None, None, "未检测到文本"

        else:
            return None, None, "OCR未检测到内容"

    except Exception as e:
        return None, None, f"识别过程中出错: {str(e)}"
