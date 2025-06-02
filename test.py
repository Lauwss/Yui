import cv2
import numpy as np
from paddleocr import PaddleOCR
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import os

# 检查图像是否有效
def is_valid_image(img):
    """检查图像是否有效"""
    return img is not None and img.size > 0

# 图像预处理
def preprocess_image(image):
    """图像预处理：灰度化、对比度增强、锐化"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.equalizeHist(enhanced)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    return result

# 检测和校正倾斜车牌区域
def correct_tilted_plate(image):
    """检测并校正倾斜的车牌区域，返回校正后的图像和车牌区域坐标"""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    edges = cv2.Canny(blurred, 20, 150)
    combined_edges = cv2.bitwise_or(thresh, edges)
    kernel = np.ones((3, 3), np.uint8)
    combined_edges = cv2.dilate(combined_edges, kernel, iterations=2)
    combined_edges = cv2.erode(combined_edges, kernel, iterations=1)

    border = 10
    combined_edges[:border, :] = 0
    combined_edges[-border:, :] = 0
    combined_edges[:, :border] = 0
    combined_edges[:, -border:] = 0

    contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

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

    if plate_contour is None:
        return image, None

    pts = plate_contour.reshape(-1, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    width = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])))
    height = int(max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2])))
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))

    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_warped = clahe.apply(gray_warped)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_warped = cv2.filter2D(enhanced_warped, -1, kernel)
    warped = cv2.cvtColor(sharpened_warped, cv2.COLOR_GRAY2BGR)

    return warped, plate_contour

# 初始化 PaddleOCR
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="ch",
    use_gpu=True,
    det_db_thresh=0.01,
    det_db_box_thresh=0.03,
    det_db_unclip_ratio=5.0,
    det_limit_side_len=2000,
    drop_score=0.5,
    show_log=False
)

# GUI 和视频处理类
class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("实时车牌识别")
        self.running = False
        self.cap = None

        # GUI 组件
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)
        self.result_label = tk.Label(root, text="车牌号: 未检测到", font=("Arial", 14))
        self.result_label.pack(pady=10)
        self.start_button = tk.Button(root, text="开始识别", command=self.start_recognition)
        self.start_button.pack(pady=5)
        self.stop_button = tk.Button(root, text="停止识别", command=self.stop_recognition, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        # 视频处理线程
        self.thread = None
        self.last_plate = ""
        self.last_update = 0

    def start_recognition(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)  # 使用默认摄像头
            if not self.cap.isOpened():
                self.result_label.config(text="错误: 无法打开摄像头")
                self.running = False
                return
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.thread = threading.Thread(target=self.process_video)
            self.thread.daemon = True
            self.thread.start()

    def stop_recognition(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.result_label.config(text="车牌号: 未检测到")
        cv2.destroyAllWindows()

    def process_video(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # 预处理
            processed_frame = preprocess_image(frame)
            output_frame = frame.copy()

            # 检测并校正车牌
            corrected_plate, plate_contour = correct_tilted_plate(processed_frame)

            # OCR 识别
            plate_text = "未检测到"
            if plate_contour is not None:
                cv2.imwrite("temp_plate.jpg", corrected_plate)
                result = ocr.ocr("temp_plate.jpg", cls=True)
                if result and result[0]:
                    best_line = max(result[0], key=lambda x: x[1][1])
                    plate_text = best_line[1][0]
                    score = best_line[1][1]
                    box = best_line[0]
                    if score > 0.8 and 6 <= len(plate_text.strip()) <= 9:
                        # 绘制车牌框
                        pts = np.array(box, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(output_frame, [pts], True, (0, 255, 0), 2)
                        cv2.polylines(output_frame, [plate_contour], True, (255, 0, 0), 2)
                        cv2.putText(
                            output_frame,
                            f"{plate_text} ({score:.2f})",
                            (int(box[0][0]), int(box[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )
            else:
                result = ocr.ocr(processed_frame, cls=True)
                if result and result[0]:
                    best_line = max(result[0], key=lambda x: x[1][1])
                    plate_text = best_line[1][0]
                    score = best_line[1][1]
                    box = best_line[0]
                    if score > 0.8 and 6 <= len(plate_text.strip()) <= 9:
                        pts = np.array(box, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(output_frame, [pts], True, (0, 255, 0), 2)
                        cv2.putText(
                            output_frame,
                            f"{plate_text} ({score:.2f})",
                            (int(box[0][0]), int(box[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

            # 更新 GUI
            current_time = time.time()
            if plate_text != self.last_plate and current_time - self.last_update > 1:
                self.result_label.config(text=f"车牌号: {plate_text}")
                self.last_plate = plate_text
                self.last_update = current_time

            # 显示视频帧
            frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            time.sleep(0.1)  # 控制帧率

        self.stop_recognition()

    def __del__(self):
        self.stop_recognition()

# 主程序
if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()