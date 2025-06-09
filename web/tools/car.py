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

def locate_license_plate(img, target_w=6000, target_h=1000):  # Renamed w, h to target_w, target_h for clarity
    """
    对输入图像进行车牌区域定位 (using user's specific morphology)
    :param img: 输入的图像 (BGR)
    :param target_w: 处理后图像的宽度
    :param target_h: 处理后图像的高度
    :return: (透视变换后的车牌区域图像, 检测到的角点) 或 (原始图像, None)
    """
    img_for_drawing = img.copy()  # Create a copy for drawing points
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义颜色阈值范围 (H: 0-180, S: 0-255, V: 0-255)
    # 蓝色阈值（适用于天空、车牌等）
    lower_blue = np.array([100, 80, 80])
    upper_blue = np.array([140, 255, 255])

    # 绿色阈值（适用于植物、交通标志等）
    lower_green = np.array([50, 40, 40])
    upper_green = np.array([85, 255, 255])

    # 生成颜色掩膜
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 形态学操作（去除噪声）
    kernel = np.ones((5, 5), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=0)
    dilate_kernel = np.ones((3, 3), np.uint8)
    mask_green = cv2.dilate(mask_green, dilate_kernel, iterations=10)
    combined_mask = cv2.bitwise_or(mask_blue, mask_green)


    # Canny 边缘检测 (using adaptive thresholding from your first example, as it's generally better)
    v = np.median(combined_mask )
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(combined_mask, lower, upper)
    # cv2.imwrite('../edge.jpg', edges) # Keep for debugging if needed

    # --- MORPHOLOGY AS PER YOUR SPECIFIC CODE ---
    # 膨胀操作
    dilate_kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, dilate_kernel, iterations=4)
    cv2.imwrite('../detailed.jpg', dilated) # Keep for debugging if needed

    # 使用腐蚀操作去除细小噪点
    erode_kernel = np.ones((3, 3), np.uint8)  # kernel was just 'kernel' before, made it explicit
    eroded = cv2.erode(dilated, erode_kernel, iterations=2)
    # --- END OF SPECIFIC MORPHOLOGY ---

    # 查找轮廓 (on the 'eroded' image as per your code)
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_contour_approx = None  # Store the approximated contour
    detected_raw_points = None  # To store the sorted points before perspective transform

    for contour in contours:
        # 计算轮廓周长
        peri = cv2.arcLength(contour, True)
        # 多边形逼近
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)  # Epsilon from your code
        # 计算轮廓面积
        area = cv2.contourArea(approx)
        if area > max_area and len(approx) == 4:
            # Basic aspect ratio check to avoid very thin or very fat rectangles if desired
            # (x,y,w_rect,h_rect) = cv2.boundingRect(approx)
            # aspect_ratio = w_rect / float(h_rect)
            # if 0.2 < aspect_ratio < 5.0: # Example, adjust as needed or remove
            max_area = area
            max_contour_approx = approx

    if max_contour_approx is not None:
        # 修正顶点顺序 (using your sorting method)
        # Reshape to (4, 2)
        current_points = max_contour_approx.reshape(-1, 2)

        # Sort by y-coordinate to separate top and bottom points
        sorted_by_y = sorted(current_points, key=lambda point: point[1])

        # Sort the top two points by x-coordinate
        top_two = sorted(sorted_by_y[:2], key=lambda point: point[0])

        # Sort the bottom two points by x-coordinate
        # Note: for [TL, TR, BL, BR] order in dst_points, bottom_two needs to be [BL, BR]
        bottom_two = sorted(sorted_by_y[2:], key=lambda point: point[0])

        # Combine to get [TL, TR, BL, BR] order
        # This order matches your dst_points: [[0,0], [W,0], [0,H], [W,H]]
        # which implies TL, TR, BL, BR if H is positive down
        detected_raw_points = np.array(top_two + bottom_two, dtype=np.float32)

        # 定义目标顶点
        # Your dst_points order: [[0,0], [target_w,0], [0,target_h], [target_w,target_h]]
        # This corresponds to: Top-Left, Top-Right, Bottom-Left, Bottom-Right
        dst_points = np.array([
            [0, 0],
            [target_w - 1, 0],  # -1 for 0-based indexing if target_w/h are pixel counts
            [0, target_h - 1],
            [target_w - 1, target_h - 1]
        ], dtype=np.float32)

        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(detected_raw_points, dst_points)
        # 进行透视变换
        warped = cv2.warpPerspective(img, matrix, (target_w, target_h))
        return warped
    # else:
    #     return img# Return warped, points, and image for drawing

def detect_colors(img, visualize=True):
    """
    检测图像中的蓝色和绿色区域
    :param img: 输入图像 (BGR格式)
    :param visualize: 是否可视化结果
    :return: 包含蓝色和绿色区域掩膜的字典
    """
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义颜色阈值范围 (H: 0-180, S: 0-255, V: 0-255)
    # 蓝色阈值（适用于天空、车牌等）
    lower_blue = np.array([100, 80, 80])
    upper_blue = np.array([140, 255, 255])

    # 绿色阈值（适用于植物、交通标志等）
    lower_green = np.array([50, 40, 40])
    upper_green = np.array([85, 255, 255])

    # 生成颜色掩膜
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 形态学操作（去除噪声）
    kernel = np.ones((5, 5), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=0)
    dilate_kernel = np.ones((3, 3), np.uint8)
    mask_green= cv2.dilate(mask_green, dilate_kernel, iterations=10)
    combined_mask = cv2.bitwise_or( mask_blue,  mask_green)
    cv2.imwrite("locate/blue_masks.jpg", combined_mask)
    # 查找区域的轮廓
    contours, _ = cv2.findContours(combined_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("未找到蓝色或绿色区域，请检查图像或调整颜色范围。")
        return

    # 找到最大的区域，假设为车牌区域
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 增加外围区域
    padding = 50  # 统一扩展量（像素）
    height, width = img.shape[:2]

    # 获取原始边界框坐标（x, y为左上角坐标，w, h为宽高）
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 计算均匀扩展后的边界框
    x1 = max(0, x - padding)  # 左侧扩展 padding 像素（向左移动padding）
    y1 = max(0, y - padding)  # 上侧扩展 padding 像素（向上移动padding）
    x2 = min(width, x + w + padding)  # 右侧扩展 padding 像素（原宽度w + padding）
    y2 = min(height, y + h + padding)  # 下侧扩展 padding 像素（原高度h + padding）

    license_plate_area_1= img[y1:y2, x1:x2]

    return      license_plate_area_1

def preprocess_image(image):
    """图像预处理：灰度化、对比度增强、锐化"""
    gray = cv2.cvtColor(  image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('eaxm.jpg',image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("preprocessed_image.jpg", result)
    return result

# plt 显示彩色图片
def plt_show0(img):
    """显示彩色图像"""
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()

# 初始化 PaddleOCR（初次识别，使用优化参数）

ocr_primary = PaddleOCR(
    use_textline_orientation=True,     # 代替 use_angle_cls
    lang="ch",
    text_det_thresh=0.03,              # 代替 det_db_thresh
    text_det_box_thresh=0.05,          # 代替 det_db_box_thresh
    text_det_unclip_ratio=5.0,         # 代替 det_db_unclip_ratio
    text_det_limit_side_len=2000       # 代替 det_limit_side_len
)

ocr_secondary = PaddleOCR(
    use_textline_orientation=True,
    lang="ch"
)


# 简单 OCR 函数（基于 ocr_demo）
def simple_ocr(image_path):
    """对图像执行简单 OCR 识别，返回文本和置信度"""
    try:
        result = ocr_secondary.predict(image_path)
        if result and result[0]:
            # 选择置信度最高的文本
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
# 车牌识别主函数
# try:
#     # 加载图像
#     image_path = os.path.abspath(os.path.join(project_root, '..', 'static', 'uploads', image['filename']))
#     image = load_image(image_path)
#     # 预处理图像
#     image_detect = detect_colors(image)
#     exam=locate_license_plate(image_detect)
#     cv2.imwrite('exam.png',exam)
#     image=preprocess_image(exam)
#     output_image = image.copy()

#     # 第一次运行 PaddleOCR（检测文本框）
#     temp_path = 'exam.png'
#     result =ocr_secondary.predict(image_path)

#     # 处理识别结果
#     license_plate = []
#     boxes = []
#     scores = []
#     if result and len(result) > 0 and len(result[0]) > 0:
#         for idx, line in enumerate(result[0]):
#             if len(line) < 2 or len(line[1]) < 2:
#                 continue
#             box = line[0]
#             text = line[1][0] if line[1][0] else ""
#             score = line[1][1] if isinstance(line[1][1], (float, int)) else 0.0
#             print(f"初次识别 文本 {idx+1}: {text}, 置信度: {score:.2f}, 坐标: {box}")

#             x_min = max(0, int(min([p[0] for p in box])) - 5)
#             x_max = min(image.shape[1], int(max([p[0] for p in box])) + 5)
#             y_min = max(0, int(min([p[1] for p in box])) - 5)
#             y_max = min(image.shape[0], int(max([p[1] for p in box])) + 5)
#             text_image = image[y_min:y_max, x_min:x_max]
#             text_image_path = f"text_{idx+1}.jpg"
#             cv2.imwrite(text_image_path, text_image)

#             re_text, re_score = simple_ocr(text_image_path)
#             if re_text and isinstance(re_text, str) and len(re_text) > 0:
#                 license_plate.append(re_text)
#                 boxes.append(box)
#                 scores.append(re_score)
#             else:
#                 license_plate.append(text)
#                 boxes.append(box)
#                 scores.append(score)

#         plate_text = ""
#         if license_plate:
#             print(f"候选车牌文本: {license_plate}")
#             candidates = [(text, score, box) for text, score, box in zip(license_plate, scores, boxes)
#                         if 6 <= len(text.strip()) <= 9]
#             if candidates:
#                 best_text, best_score, best_box = max(candidates, key=lambda x: x[1])
#                 plate_text = best_text
#                 print(f"选择置信度最高的车牌: {plate_text}, 置信度: {best_score:.2f}")
#                 boxes = [best_box]
#                 license_plate = [best_text]
#             else:
#                 sorted_results = sorted(zip(boxes, license_plate, scores), key=lambda x: x[0][0][0])
#                 license_plate = []
#                 boxes = []
#                 merged_text = ""
#                 prev_x_max = -float('inf')
#                 prev_box = None

#                 for box, text, score in sorted_results:
#                     x_min = min([p[0] for p in box])
#                     x_max = max([p[0] for p in box])
#                     if x_min - prev_x_max < 100:
#                         merged_text += text
#                     else:
#                         if merged_text:
#                             license_plate.append(merged_text)
#                             boxes.append(prev_box if prev_box else box)
#                         merged_text = text
#                     prev_x_max = x_max
#                     prev_box = box

#                 if merged_text:
#                     license_plate.append(merged_text)
#                     boxes.append(prev_box if prev_box else box)

#                 plate_text = ''.join(license_plate)  # 不强制截断
#                 print(f"合并后车牌文本: {plate_text}")
#                 if not plate_text:
#                     plate_text = "无"
#         else:
#             print("未检测到车牌文本")
#             plate_text = "无"
#     else:
#         print("PaddleOCR 未检测到任何文本")
#         plate_text = "无"


#     # 可视化结果
#     for box, text in zip(boxes, license_plate):
#         # 绘制文本框
#         pts = np.array(box, np.int32).reshape((-1, 1, 2))
#         cv2.polylines(output_image, [pts], True, (0, 255, 0), 2)
#         # 绘制文本
#         cv2.putText(
#             output_image,
#             text,
#             (int(box[0][0]), int(box[0][1]) - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 0, 255),
#             2
#         )

#     # 保存和显示结果
#     cv2.imwrite("output_image.jpg", output_image)
#     print("框出车牌的图像已保存为 output_image.jpg")
#     plt_show0(output_image)

#     # 输出最终结果
#     print(f"最终车牌号码: {plate_text}")

#     # 保存识别结果到文件（基于 ocr_demo）
#     with open("result.txt", "w", encoding="utf-8") as f:
#         for text in license_plate:
#             f.write(f"{text}\n")

# except Exception as e:
#     print(f"处理过程中发生错误: {str(e)}")