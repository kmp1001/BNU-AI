

import os
import random
import re
import shutil
import time

import cv2
import numpy as np

from PIL import ImageFont, ImageDraw, Image
show_flag = False

# 指定字体路径（请根据操作系统调整路径）
font_path = "C:/Windows/Fonts/simhei.ttf"  # Windows 黑体
font_pil = ImageFont.truetype(font_path, 36)

def cv_show(win_name, img, origin_img=None, size=(900, 600)):
    """显示图像，可选定尺寸，窗口等待键盘输入后关闭"""
    if origin_img is None:
        img_resized = cv2.resize(img, size)
        cv2.imshow(win_name, img_resized)
        cv2.waitKey(1)
    else:
        overlay = origin_img.copy()
        overlay[img == 255] = (0, 255, 0)
        img = cv2.addWeighted(overlay, 0.7, origin_img, 0.3, 0)
        img_resized = cv2.resize(img, size)
        cv2.imshow(win_name, img_resized)
        cv2.waitKey(1)


def init_template_dict():
    template_filedir = 'video_template/video_template'
    temp_path = 'temp.bmp'
    template_dict = {key: [] for key in os.listdir(template_filedir)}
    for root, dirs, files in os.walk(template_filedir):
        key = os.path.basename(root)
        for file in files:
            image_path = os.path.join(root, file)
            shutil.copy(image_path, temp_path)
            temp_image = cv2.imread(temp_path)
            temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(temp_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            thresh = cv2.medianBlur(thresh, 3)
            template_dict[key].append(thresh)
    os.remove(temp_path)
    return template_dict


class HSVDict:
    def __init__(self, low_h, high_h, low_s, high_s, low_v, high_v):
        self.low_h = low_h
        self.high_h = high_h
        self.low_s = low_s
        self.high_s = high_s
        self.low_v = low_v
        self.high_v = high_v
        self.lower = np.array([self.low_h, self.low_s, self.low_v])
        self.higher = np.array([self.high_h, self.high_s, self.high_v])

    def get_lower_range(self):
        return self.lower

    def get_higher_range(self):
        return self.higher


SOBEL_TEMPLATE = [
    np.array([-2, -1, 0, -1, 0, 1, 0, 1, 2]).reshape(3, 3),
    np.array([0, 1, 2, -1, 0, 1, -2, -1, 0]).reshape(3, 3),
    np.array([1, 2, 1, 0, 0, 0, -1, -2, -1]).reshape(3, 3),
    np.array([2, 1, 0, 1, 0, -1, 0, -1, -2]).reshape(3, 3),
    np.array([1, 0, -1, 2, 0, -2, 1, 0, -1]).reshape(3, 3),
    np.array([0, -1, -2, 1, 0, -1, 2, 1, 0]).reshape(3, 3),
    np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1]).reshape(3, 3),
    np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape(3, 3)
]

def extended_sobel(gray_frame):
    result_frame = np.zeros_like(gray_frame)
    for sobel_operator in SOBEL_TEMPLATE:
        filtered = cv2.filter2D(gray_frame, cv2.CV_32F, sobel_operator)
        result_frame = np.maximum(result_frame, np.abs(filtered))
    result_frame = cv2.normalize(result_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return result_frame


blue = [
    np.array([0, 0, 0]),
    np.array([180, 159, 256])
]

def order_points(pts):
    '''对4点进行排序，下标0、1、2、3分别是左上、右上、右下、左下'''
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspectiveTransform(pntsOrded, img):
    tl, tr, br, bl = pntsOrded

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(pntsOrded, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

def get_contours(gray_image):
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours

def get_contour_by_ratio(gray_image: np.ndarray, ratio=0.4):
    height, width = gray_image.shape[:2]
    area = height * width
    result = list()
    contours = get_contours(gray_image)
    for contour in contours:
        contour_ratio = cv2.contourArea(contour) / area
        if contour_ratio < ratio:
            break
        result.append(contour)
    return result

def mask_image_by_contour_ratio(gray_image, ratio):
    mask_contour = get_contour_by_ratio(gray_image, ratio)
    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, mask_contour, -1, 255, thickness=cv2.FILLED)
    result_image = cv2.bitwise_and(mask, gray_image)
    return result_image

def include_chinese(string: str):
    pattern = re.compile(r'[\u4e00-\u9fa5+]')
    match = pattern.search(string)
    return match is not None

def match_character(char_img, templates, location=0):
    """
    使用模板匹配对单个字符图像进行识别。
    遍历所有模板，选择匹配得分最高的字符。
    """
    best_match = None
    best_score = -1
    for index, (char, t_list) in enumerate(templates.items()):
        # 按位置过滤中英文模板
        if location == 0:
            if not include_chinese(char):
                continue
        else:
            if include_chinese(char):
                continue

        for t in t_list:
            w = char_img.shape[1]
            h = char_img.shape[0]
            # 将模板尺寸调整为与候选字符图像一致
            resized_t = cv2.resize(t, (w, h))
            # 这里只给了一个角度遍历示例，如有需要可加大角度范围或更精细
            for angle in range(0, 1, 5):
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                templateR = cv2.warpAffine(resized_t, M, (w, h))

                result = cv2.matchTemplate(char_img, templateR, cv2.TM_CCOEFF_NORMED)
                score = result.max()
                if score > best_score:
                    best_score = score
                    best_match = char
    return best_match, best_score


# 初始化模板字典
template_dict = init_template_dict()


def process_frame(frame):
    """
    对单帧图像进行处理。
    这里除了识别以外，为了可视化，还会对检测到的车牌区域在原图上画一个红色外接矩形，
    并返回这张“处理后”的帧用于保存到视频中。
    """
    # === 1. 常规预处理 ===
    frame_blur = cv2.GaussianBlur(frame, (3, 3), 0)
    hsv_frame = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, blue[0], blue[1])
    ycrcb_frame = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2YCrCb)
    # 让蓝色区域的Cr分量更突出
    ycrcb_frame[mask == 255][1] = 255
    y_frame, cr_frame, cb_frame = cv2.split(ycrcb_frame)

    # 二值化 + 形态学
    _, binary = cv2.threshold(cr_frame, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel)

    # === 2. 轮廓检测 ===
    sobel_frame = extended_sobel(binary)
    contours, _ = cv2.findContours(sobel_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_contour = list()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = w / float(h)
        if 2 <= aspect_ratio <= 4:
            if 1000 <= area < 5000:
                if area / (w * h) > 0.5:
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    # 要求边数合适
                    if 4 <= len(approx) <= 10:
                        result_contour.append(contour)

    # === 3. 从检测到的车牌轮廓中透视变换，分割字符并识别 ===
    green_plate_list = list()
    blue_plate_list = list()

    for contour in result_contour:
        rotated_rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rotated_rect)
        pts = order_points(box)
        temp_frame = perspectiveTransform(pts, frame)
        temp_gray = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY)
        _, temp_binary = cv2.threshold(temp_gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        b, g, r = map(np.sum, cv2.split(temp_frame))
        gg = g / (b + g + r)
        bb = b / (b + g + r)
        resize_frame = cv2.resize(temp_frame, (76, 22))

        # 根据简单的颜色占比判断绿色车牌 or 蓝色车牌
        if gg > 0.34:
            green_plate_list.append((resize_frame, box))
        elif 0.49 > bb > 0.45 and gg > 0.2945:
            blue_plate_list.append((resize_frame, box))

    # 开始拆字符
    start_b = 2
    width_b = 10
    offset_b = 10
    space_b = 4

    character_list = []
    for (blue_plate, box) in blue_plate_list:
        temp = list()
        for i in range(7):
            x0 = start_b + i * offset_b
            x1 = width_b + i * offset_b
            if i >= 2:
                x0 += space_b
                x1 += space_b
            temp.append(blue_plate[:, x0:x1])
        character_list.append(temp)

        # 在原图上画出框(可视化)
        int_box = np.int0(box)
        cv2.drawContours(frame, [int_box], 0, (0, 0, 255), 2)

    start_g = 2
    width_g = 8
    offset_g = 8
    space_g = 4

    for (green_plate, box) in green_plate_list:
        temp = list()
        for i in range(8):
            x0 = start_g + i * offset_g
            x1 = width_g + i * offset_g
            if i >= 2:
                x0 += space_g
                x1 += space_g
            temp.append(green_plate[:, x0:x1])
        character_list.append(temp)

        # 在原图上画出框(可视化)
        int_box = np.int0(box)
        cv2.drawContours(frame, [int_box], 0, (0, 0, 255), 2)

        # === 4. 逐字符匹配 ===
    number = ''
    for plate_idx, character in enumerate(character_list):
        number = ''  # 清空车牌号用于每块牌单独识别
        for location, char_img in enumerate(character):
            t_char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
            _, char_bin = cv2.threshold(t_char_img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            char, prob = match_character(char_bin, template_dict, location)
            if show_flag:
                cv_show('char', char_bin)
            number += char.upper()

        # 测试打印，或在图上写车牌号
        print("识别结果:", number)
        # 识别结果显示在画面左上角
        cv2.putText(frame, number, (20, 40 + 40 * plate_idx),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # 返回这帧标注后的画面
    return frame


# ...（main 函数无改动）...

def main():
    cap = cv2.VideoCapture('car.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('processed_result.mp4', fourcc, fps if fps > 0 else 25, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        out.write(processed_frame)

        cv2.imshow('result', processed_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
