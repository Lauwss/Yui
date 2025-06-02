import cv2
import numpy as np
from paddleocr import PaddleOCR
from matplotlib import pyplot as plt
import os

# 检查图像是否正确加载
def load_image(image_path):
    """加载图像并检查是否有效"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件 {image_path} 不存在")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图像 {image_path}")
    return img

# 图像预处理
def preprocess_image(image):
    """图像预处理：灰度化、对比度增强、锐化、二值化"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.equalizeHist(enhanced)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("preprocessed_image.jpg", result)
    cv2.imwrite("binary_image.jpg", binary)
    return result

# 改进：检测和校正倾斜车牌区域
def correct_tilted_plate(image):
    """检测并校正倾斜的车牌区域，返回校正后的图像和车牌区域坐标"""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    edges = cv2.Canny(blurred, 20, 150)  # 进一步降低低阈值
    combined_edges = cv2.bitwise_or(thresh, edges)
    kernel = np.ones((3, 3), np.uint8)
    combined_edges = cv2.dilate(combined_edges, kernel, iterations=2)
    combined_edges = cv2.erode(combined_edges, kernel, iterations=1)
    cv2.imwrite("combined_edges.jpg", combined_edges)

    border = 10
    combined_edges[:border, :] = 0
    combined_edges[-border:, :] = 0
    combined_edges[:, :border] = 0
    combined_edges[:, -border:] = 0
    cv2.imwrite("edges_no_border.jpg", combined_edges)

    contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    debug_all_contours = image.copy()
    cv2.drawContours(debug_all_contours, contours, -1, (0, 255, 0), 2)
    cv2.imwrite("all_contours.jpg", debug_all_contours)
    print("所有检测到的轮廓已保存为 all_contours.jpg")

    plate_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        min_area = (width * height) / 100
        max_area = (width * height) / 5
        if area < min_area or area > max_area:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) >= 4:
            pts = approx.reshape(-1, 2)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            width = max(np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4))
            height = max(np.linalg.norm(box[i] - box[(i + 2) % 4]) for i in range(4))
            aspect_ratio = width / height if height != 0 else 0
            if 2 <= aspect_ratio <= 6:
                x, y, w, h = cv2.boundingRect(contour)
                if x > border and y > border and (x + w) < (width - border) and (y + h) < (height - border):
                    plate_contour = approx
                    break

    # 即使找到部分轮廓，也尝试用Hough变换补充缺失边缘
    hough_applied = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            if x > border and y > border and (x + w) < (width - border) and (y + h) < (height - border):
                # 扩展ROI区域，向下扩展以包含可能的下边
                y_extend = int(h * 0.5)  # 向下扩展50%
                y_min = max(0, y - int(h * 0.1))
                y_max = min(height, y + h + y_extend)
                x_min = max(0, x - int(w * 0.1))
                x_max = min(width, x + w + int(w * 0.1))
                roi = combined_edges[y_min:y_max, x_min:x_max]
                cv2.imwrite("roi_image.jpg", roi)  # 保存ROI区域以供调试
                print("ROI区域已保存为 roi_image.jpg")

                # Hough变换检测直线
                lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=30, minLineLength=w/4, maxLineGap=20)
                if lines is not None:
                    hough_applied = True
                    # 调试：绘制检测到的直线
                    debug_lines = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(debug_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imwrite("hough_lines.jpg", debug_lines)
                    print("Hough变换检测到的直线已保存为 hough_lines.jpg")

                    # 构造四边形
                    pts = []
                    for line in lines[:4]:
                        x1, y1, x2, y2 = line[0]
                        pts.append([x1 + x_min, y1 + y_min])
                        pts.append([x2 + x_min, y2 + y_min])
                    if len(pts) >= 4:
                        pts = np.array(pts, dtype=np.float32)
                        hull = cv2.convexHull(pts)
                        approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
                        if len(approx) >= 4:
                            plate_contour = approx
                            break

    # 如果仍未找到四边形，尝试手动构造
    if plate_contour is None and contours:
        print("Hough变换未找到足够直线，尝试手动构造四边形")
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(contour)
                if x > border and y > border and (x + w) < (width - border) and (y + h) < (height - border):
                    plate_contour = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
                    break

    if plate_contour is None:
        print("仍然未检测到可能的车牌区域，直接返回原图")
        return image, None

    # 调试：绘制最终轮廓并保存
    debug_image = image.copy()
    cv2.polylines(debug_image, [plate_contour.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.imwrite("contour_image.jpg", debug_image)
    print("检测到的车牌轮廓已保存为 contour_image.jpg")

    # 获取车牌四个角点
    pts = plate_contour.reshape(-1, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # 按顺序排列角点：左上、右上、右下、左下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    # 计算目标矩形的宽高
    width = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])))
    height = int(max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2])))
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    # 透视变换
    M = cv2.getPerspectiveTransform(rect, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))

    # 增强校正后的图像
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_warped = clahe.apply(gray_warped)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_warped = cv2.filter2D(enhanced_warped, -1, kernel)
    warped = cv2.cvtColor(sharpened_warped, cv2.COLOR_GRAY2BGR)

    # 保存校正后的车牌区域
    cv2.imwrite("corrected_plate.jpg", warped)
    print("校正后的车牌区域已保存为 corrected_plate.jpg")

    return warped, plate_contour

# plt 显示彩色图片
def plt_show0(img):
    """显示彩色图像"""
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()

# 初始化 PaddleOCR（初次识别，使用优化参数）
ocr_primary = PaddleOCR(
    use_angle_cls=True,
    lang="ch",
    use_gpu=True,
    det_db_thresh=0.01,
    det_db_box_thresh=0.03,
    det_db_unclip_ratio=5.0,
    det_limit_side_len=2000,
    drop_score=0.5,
    show_log=True
)

# 初始化 PaddleOCR（二次识别，使用默认参数，类似 ocr_demo）
ocr_secondary = PaddleOCR(
    use_angle_cls=True,
    lang="ch",
    use_gpu=True,
)

# 简单 OCR 函数（基于 ocr_demo）
def simple_ocr(image_path):
    """对图像执行简单 OCR 识别，返回文本和置信度"""
    try:
        result = ocr_secondary.ocr(image_path, cls=True)
        if result and result[0]:
            best_line = max(result[0], key=lambda x: x[1][1])
            text = best_line[1][0]
            confidence = best_line[1][1]
            print(f"简单 OCR 识别 {image_path}: 文本: {text}, 置信度: {confidence:.4f}")
            return text, confidence
        else:
            print(f"简单 OCR 识别 {image_path}: 未检测到文本")
            return "", 0.0
    except Exception as e:
        print(f"简单 OCR 识别 {image_path} 失败: {str(e)}")
        return "", 0.0

# 主处理流程
try:
    # 加载图像
    image_path = 'xiede.jpeg'
    image = load_image(image_path)
    # 预处理图像
    image = preprocess_image(image)

    # 检测并校正倾斜车牌
    corrected_plate, plate_contour = correct_tilted_plate(image)
    output_image = image.copy()

    # 如果校正成功，使用校正后的车牌区域进行OCR
    if plate_contour is not None:
        corrected_image_path = "corrected_plate.jpg"
        result = ocr_primary.ocr(corrected_image_path, cls=True)
        print("对校正后的车牌区域进行OCR识别")
        if not result or not result[0]:
            print("校正后的图像未检测到车牌，尝试对原始图像进行OCR")
            result = ocr_primary.ocr(image_path, cls=True)
    else:
        result = ocr_primary.ocr(image_path, cls=True)
        print("未检测到倾斜车牌，使用原始图像进行OCR识别")

    # 处理识别结果
    license_plate = []
    boxes = []
    scores = []
    if result and result[0]:
        for idx, line in enumerate(result[0]):
            box = line[0]
            text = line[1][0]
            score = line[1][1]
            print(f"初次识别 文本 {idx+1}: {text}, 置信度: {score:.2f}, 坐标: {box}")

            if plate_contour is not None:
                license_plate.append(text)
                boxes.append(box)
                scores.append(score)
                continue

            x_min = int(min([p[0] for p in box])) - 5
            x_max = int(max([p[0] for p in box])) + 5
            y_min = int(min([p[1] for p in box])) - 5
            y_max = int(max([p[1] for p in box])) + 5
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image.shape[1], x_max)
            y_max = min(image.shape[0], y_max)
            text_image = image[y_min:y_max, x_min:x_max]
            text_image_path = f"text_{idx+1}.jpg"
            cv2.imwrite(text_image_path, text_image)

            re_text, re_score = simple_ocr(text_image_path)
            if re_text:
                license_plate.append(re_text)
                boxes.append(box)
                scores.append(re_score)
            else:
                license_plate.append(text)
                boxes.append(box)
                scores.append(score)

        plate_text = ""
        if license_plate:
            print(f"候选车牌文本: {license_plate}")
            candidates = [(text, score, box) for text, score, box in zip(license_plate, scores, boxes)
                          if 6 <= len(text.strip()) <= 9]
            if candidates:
                best_text, best_score, best_box = max(candidates, key=lambda x: x[1])
                plate_text = best_text
                print(f"选择置信度最高的车牌: {plate_text}, 置信度: {best_score:.2f}")
                boxes = [best_box]
                license_plate = [best_text]
            else:
                sorted_results = sorted(zip(boxes, license_plate, scores), key=lambda x: x[0][0][0])
                license_plate = []
                boxes = []
                merged_text = ""
                prev_x_max = -float('inf')

                for box, text, score in sorted_results:
                    x_min = min([p[0] for p in box])
                    x_max = max([p[0] for p in box])
                    if x_min - prev_x_max < 100:
                        merged_text += text
                    else:
                        if merged_text:
                            license_plate.append(merged_text)
                            boxes.append(prev_box if 'prev_box' in locals() else box)
                        merged_text = text
                    prev_x_max = x_max
                    prev_box = box

                if merged_text:
                    license_plate.append(merged_text)
                    boxes.append(prev_box if 'prev_box' in locals() else box)

                plate_text = ''.join(license_plate)[:8]
                print(f"合并后车牌文本: {plate_text}")
                if not plate_text:
                    plate_text = "无"
        else:
            print("未检测到车牌文本")
            plate_text = "无"
    else:
        print("PaddleOCR 未检测到任何文本")
        plate_text = "无"

    for box, text in zip(boxes, license_plate):
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(output_image, [pts], True, (0, 255, 0), 2)
        cv2.putText(
            output_image,
            text,
            (int(box[0][0]), int(box[0][1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    if plate_contour is not None:
        cv2.polylines(output_image, [plate_contour], True, (255, 0, 0), 2)

    cv2.imwrite("output_image.jpg", output_image)
    print("框出车牌的图像已保存为 output_image.jpg")
    plt_show0(output_image)

    print(f"最终车牌号码: {plate_text}")

    with open("result.txt", "w", encoding="utf-8") as f:
        for text in license_plate:
            f.write(f"{text}\n")

except Exception as e:
    print(f"处理过程中发生错误: {str(e)}")