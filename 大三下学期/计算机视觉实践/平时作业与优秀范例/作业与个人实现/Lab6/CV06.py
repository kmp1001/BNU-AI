import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.pyplot as plt

# 辅助函数：使用 matplotlib 显示图像
def show_image(title, image, pause_time=0):
    plt.figure()
    plt.title(title)
    if image is None or image.size == 0:
        print(f"{title} 图像为空，无法显示")
        return
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        plt.imshow(rgb_image)
    plt.axis('off')
    if pause_time > 0:
        plt.pause(pause_time)
        plt.close()
    else:
        plt.show()

# 全局变量定义
# 字符字典：前10个为数字，接下来24个为英文字母（不含 I 和 O），最后31个为中文省份简称
List = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] + \
       ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
        'Y', 'Z'] + \
       ['京', '津', '沪', '渝', '冀', '豫', '云', '辽', '黑', '湘', '皖', '鲁', '新', '苏', '浙', '赣', '鄂', '桂',
        '甘', '晋', '蒙', '陕', '吉', '闽', '贵', '粤', '青', '藏', '川', '宁', '琼']

final_result = []  # 保存最终识别出的车牌字符
img = None         # 原始图像
con = 2            # 模板匹配控制变量：0-汉字，1-字母，2-数字+字母

def Text(image, text, p, color, size):
    """
    在图像上绘制中文文字
    """
    cv2_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("./simhei.ttf", size, encoding="utf-8")
    draw.text((p[0] - 60, p[1] - 20), text, fill=color, font=font)
    cv2_result = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
    return cv2_result

def Show_Result_Words(index):
    print("识别结果字符：", List[index])
    final_result.append(List[index])
    print("当前车牌识别结果：", final_result)

def Show_Result_Image(plate_img):
    """
    在车牌图像上标注识别出的完整车牌信息
    """
    result_img = Text(plate_img.copy(), ''.join(final_result), (10, 30), (255, 0, 0), 20)
    show_image("Final Result", result_img, pause_time=0)

def Template_Match(image):
    """
    对单字符图像进行模板匹配，匹配分数最高的模板对应的字符即为识别结果
    :param image: 单字符图像（BGR格式）
    """
    global List, final_result, con
    if image is None or image.size == 0:
        print("Template_Match 收到空图像，跳过该字符。")
        return
    show_image("Template_Match - Input Segment", image, pause_time=0.3)
    best_score = []
    index = 0

    if con == 0:
        # 省份汉字模板：文件夹索引 34~64
        for i in range(34, 65):
            score = []
            folderPath = os.path.join('Template', List[i])
            if not os.path.isdir(folderPath):
                continue
            for filename in os.listdir(folderPath):
                path = os.path.join(folderPath, filename)
                template = cv.imdecode(np.fromfile(path, dtype=np.uint8), 1)
                if template is None:
                    continue
                gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
                ret, template_bin = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
                h, w = template_bin.shape
                try:
                    resized_image = cv.resize(image, (w, h), interpolation=cv.INTER_CUBIC)
                except Exception as e:
                    print("resize 错误:", e)
                    continue
                # show_image("Resized Segment", resized_image, pause_time=0.1)
                # show_image("Template", template_bin, pause_time=0.1)
                resized_gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
                result = cv.matchTemplate(resized_gray, template_bin, cv.TM_CCOEFF)
                score.append(result[0][0])
            best_score.append(max(score) if score else -np.inf)
        if best_score:
            index = best_score.index(max(best_score)) + 34

    elif con == 1:
        # 字母模板：文件夹索引 10~33
        for i in range(10, 34):
            score = []
            folderPath = os.path.join('Template', List[i])
            if not os.path.isdir(folderPath):
                continue
            for filename in os.listdir(folderPath):
                path = os.path.join(folderPath, filename)
                template = cv.imdecode(np.fromfile(path, dtype=np.uint8), 1)
                if template is None:
                    continue
                gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
                ret, template_bin = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
                h, w = template_bin.shape
                try:
                    resized_image = cv.resize(image, (w, h), interpolation=cv.INTER_CUBIC)
                except Exception as e:
                    print("resize 错误:", e)
                    continue
                # show_image("Resized Segment", resized_image, pause_time=0.1)
                # show_image("Template", template_bin, pause_time=0.1)
                resized_gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
                result = cv.matchTemplate(resized_gray, template_bin, cv.TM_CCOEFF)
                score.append(result[0][0])
            best_score.append(max(score) if score else -np.inf)
        if best_score:
            index = best_score.index(max(best_score)) + 10

    else:
        # 数字或字母模板：文件夹索引 0~33
        for i in range(0, 34):
            score = []
            folderPath = os.path.join('Template', List[i])
            if not os.path.isdir(folderPath):
                continue
            for filename in os.listdir(folderPath):
                path = os.path.join(folderPath, filename)
                template = cv.imdecode(np.fromfile(path, dtype=np.uint8), 1)
                if template is None:
                    continue
                gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
                ret, template_bin = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
                h, w = template_bin.shape
                try:
                    resized_image = cv.resize(image, (w, h), interpolation=cv.INTER_CUBIC)
                except Exception as e:
                    print("resize 错误:", e)
                    continue
                # show_image("Resized Segment", resized_image, pause_time=0.1)
                # show_image("Template", template_bin, pause_time=0.1)
                resized_gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
                result = cv.matchTemplate(resized_gray, template_bin, cv.TM_CCOEFF)
                score.append(result[0][0])
            best_score.append(max(score) if score else -np.inf)
        if best_score:
            index = best_score.index(max(best_score))
    Show_Result_Words(index)

def main():
    global img, final_result, con
    image_path = "1.png"  # 请修改为你的图像路径
    img = cv.imread(image_path)
    if img is None:
        print("错误：无法读取图片", image_path)
        return

    # 显示原始图像
    show_image("Original Image", img, pause_time=0.3)

    # 交互式选择车牌区域
    print("请框选车牌区域...")
    plate_rect = cv.selectROI("Select License Plate", img, fromCenter=False, showCrosshair=True)
    x, y, w, h = plate_rect
    if w == 0 or h == 0:
        print("未选定有效车牌区域！")
        return
    plate_img = img[y:y+h, x:x+w]
    show_image("Selected Plate", plate_img, pause_time=0.5)

    # 交互式选择车牌中7个字符区域
    segments = []
    print("请分别框出车牌的7个字符区域...")
    for i in range(7):
        roi = cv.selectROI(f"Select Character {i+1}", plate_img, fromCenter=False, showCrosshair=True)
        x1, y1, w1, h1 = roi
        if w1 == 0 or h1 == 0:
            print(f"Character {i+1} ROI 未选定，跳过")
        else:
            char_img = plate_img[y1:y1+h1, x1:x1+w1]
            segments.append(char_img)
            show_image(f"Segment {i+1}", char_img, pause_time=0.3)
    cv.destroyAllWindows()  # 关闭交互窗口

    final_result = []
    # 对每个选定字符区域进行模板匹配识别
    for i, seg in enumerate(segments):
        if i == 0:
            con = 0  # 第一字符：省份汉字
        elif i == 1:
            con = 1  # 第二字符：字母
        else:
            con = 2  # 后续字符：数字或字母
        Template_Match(seg)

    # 显示最终结果
    Show_Result_Image(plate_img)

if __name__ == '__main__':
    main()
