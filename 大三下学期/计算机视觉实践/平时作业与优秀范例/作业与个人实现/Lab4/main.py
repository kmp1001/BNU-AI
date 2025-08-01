import traceback

import cv2
import numpy as np

DEBUG = True
SHOW_JUDGE = True
VIDEO_MODE = False
ANSWER_FILE = 'Answers.txt'
# 答题区域配置参数
ANSWER_CONFIG = {
    # 题目区域整体起始坐标
    "start": np.array([51, 315]),
    # 单个选项尺寸
    "option_size": np.array([30, 15]),
    # 每道题目水平方向的步长（选项间距）
    "option_step_x": 48,
    # 每道题目垂直方向的步长（选项行高）
    "option_step_y": 28,
    # 每个题块的水平偏移
    "block_offset_x": 48 * 3 + 104,
    # 每个题块的垂直偏移
    "block_offset_y": 28 * 4 + 38
}

# 学号区域配置参数
ID_CONFIG = {
    # 学号区域起始坐标
    "start": np.array([202, 21]),
    # 单个学号选项尺寸
    "option_size": np.array([30, 15]),
    # 每行学号的垂直步长
    "row_step": 26.5,
    # 每列学号的水平步长
    "col_step": 44.5
}


class GradeWriter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.grade_dict = dict()

    def initialize_grade_file(self):
        """初始化成绩文件，写入表头"""
        with open(self.file_path, 'w') as f:
            f.write('Student_id\tGrade\n')
        if DEBUG:
            print("Initialized Grade file")

    def append_grade(self, student_id, grade):
        """将学生学号和成绩追加写入成绩文件"""
        self.grade_dict[student_id] = grade
        if DEBUG:
            print("Written Grade")

    def flush_grade_file(self):
        self.initialize_grade_file()
        with open(self.file_path, 'a') as f:
            for key, value in self.grade_dict.items():
                f.write(f'{key}\t{value:.2f}\n')


def cv2_show(win_name, img, size=(900, 600)):
    """显示图像，可选定尺寸，窗口等待键盘输入后关闭"""
    img_resized = cv2.resize(img, size)
    cv2.imshow(win_name, img_resized)
    if not VIDEO_MODE:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_image(image):
    """读取图像，转换为灰度，并进行高斯模糊预处理"""
    if image is None:
        raise ValueError("无法读取图像，请检查路径或文件格式。")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # processed = cv2.GaussianBlur(gray, (2, 2), 1)
    processed = gray
    if DEBUG:
        cv2.imshow("Original", image)
        cv2.imshow("Processed", processed)
    return image, processed


def order_points(pts):
    '''对4点进行排序，下标0、1、2、3分别是左上、右上、右下、左下'''
    rect = np.zeros((4, 2), dtype="float32")
    # 计算左上和右下
    s = pts.sum(axis=1)  # 每个点的x和y相加，左上是和最小的，右下是和最大的。
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算右上和左下
    diff = np.diff(pts, axis=1)  # 每个点的y-x，右上最小，左下最大。
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 透视变换
def perspectiveTransform(pntsOrded, img):
    tl, tr, br, bl = pntsOrded
    # 计算输入的w和h值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(pntsOrded, dst)
    if DEBUG:
        print("变换矩阵：", M)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return warped


# 透视逆变换
def invPerspectiveTransform(pntsOrded, imgSrc, imgDst):
    height, width = imgSrc.shape[:2]

    # 变换前对应坐标位置
    src = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")

    heightDst, widthDst = imgDst.shape[:2]
    # 计算变换矩阵并进行变换
    M = cv2.getPerspectiveTransform(src, pntsOrded)
    warped = cv2.warpPerspective(imgSrc, M, (widthDst, heightDst))
    return warped


def get_rectangle_corners(contour):
    """对轮廓进行多边形近似，并返回排序后的四个角点"""
    arc = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * arc, True).reshape(-1, 2)
    if approx.shape[0] != 4:
        raise ValueError("轮廓近似得到的角点数不是4个。")
    return order_points(approx)


def get_contours(thresh):
    """查找二值图像中的轮廓，并按面积降序排列"""
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)


def extract_contours_by_idx(thresh, idx):
    """从排序后的轮廓中提取指定索引的轮廓"""
    contours = get_contours(thresh)
    if isinstance(idx, int):
        return contours[idx]
    elif isinstance(idx, list):
        return [contours[i] for i in idx]
    else:
        raise ValueError("不支持的索引类型")


def extract_id_and_question_contours(thresh):
    """排序后第0个为答题区域，第1个为学号区域"""
    contours = extract_contours_by_idx(thresh, [0, 1])
    return contours[1], contours[0]


def transform_regions(img, thresh, pts, clip=-1):
    """对指定区域进行透视变换，可选剪裁边缘"""
    trans_img = perspectiveTransform(pts, img)
    trans_thresh = perspectiveTransform(pts, thresh)
    if clip > 0:
        trans_img = trans_img[clip:-clip, clip:-clip]
        trans_thresh = trans_thresh[clip:-clip, clip:-clip]
    return trans_img, trans_thresh


def process_questions(color_img, thresh_img, thresh=182):
    """
    处理答题区域：
      - 读取答案文件，按预设版面判断填涂情况；
      - 在 color_img 上绘制结果（正确填涂用绿色、错误填涂用红色，未填涂但应填用黄色，蓝色为检测框）；
      - 返回绘制后的图像和得分（正确率）。
    """
    with open(ANSWER_FILE, 'r') as f:
        answers = f.readline().split()
    answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    _, binary = cv2.threshold(thresh_img, thresh, 255, cv2.THRESH_BINARY)
    correct_count = 0
    total = len(answers)

    start = ANSWER_CONFIG["start"].copy()
    option_size = ANSWER_CONFIG["option_size"].copy()
    step_x = ANSWER_CONFIG["option_step_x"]
    step_y = ANSWER_CONFIG["option_step_y"]
    block_dx = ANSWER_CONFIG["block_offset_x"]
    block_dy = ANSWER_CONFIG["block_offset_y"]

    # 题块分为4部分（最后一块只有1列，其余为3列）
    for block in range(4):
        region_origin = start.copy()
        region_origin[1] += block_dy * block
        cols = 3 if block != 3 else 1
        for col in range(cols):
            for row in range(5):
                option_origin = region_origin.copy()
                option_origin[1] += step_y * row
                option_values = []
                q_idx = 15 * block + 5 * col + row
                correct_option = answer_map[answers[q_idx]]
                for option in range(4):
                    x, y = option_origin
                    bubble = binary[y:y + option_size[1],x:x + option_size[0]]
                    bubble = cv2.bitwise_not(bubble)
                    ratio = cv2.countNonZero(bubble) / (option_size[0] * option_size[1])
                    option_values.append(ratio)
                    if SHOW_JUDGE:
                        if ratio > 0.5:
                            color = (0, 255, 0) if correct_option == option else (0, 0, 255)
                        else:
                            color = (0, 255, 255) if correct_option == option else (255, 0, 0)
                        cv2.rectangle(color_img, (x, y), (x + option_size[0], y + option_size[1]), color, -1)
                    option_origin[0] += step_x
                if len([i for i, v in enumerate(option_values) if v > 0.5]) == 1 and option_values.index(
                        max(option_values)) == correct_option:
                    correct_count += 1
            region_origin[0] += block_dx
    score = correct_count / total * 100
    if DEBUG and SHOW_JUDGE:
        cv2_show('Question Image', color_img)
    return score


def process_id(id_img, id_thresh):
    """
    处理学号区域：
      - 根据填涂情况判断每一位数字，选择填涂率最高的数字；
      - 返回拼接得到的学号字符串。
    """
    _, binary = cv2.threshold(id_thresh, 182, 255, cv2.THRESH_BINARY)
    student_id = ''
    start = ID_CONFIG["start"].copy()
    option_size = ID_CONFIG["option_size"].copy()
    row_step = ID_CONFIG["row_step"]
    col_step = ID_CONFIG["col_step"]
    for digit in range(12):
        col_origin = start.copy()
        col_origin[0] += col_step * digit
        digit_values = []
        for num in range(10):
            x, y = col_origin
            bubble = binary[y:y + option_size[1], x:x + option_size[0]]
            bubble = cv2.bitwise_not(bubble)
            ratio = cv2.countNonZero(bubble) / (option_size[0] * option_size[1])
            if SHOW_JUDGE:
                if ratio > 0.5:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                cv2.rectangle(id_img, (x, y), (x + option_size[0], y + option_size[1]), color, -1)
            digit_values.append(ratio)
            col_origin[1] += row_step
        student_id += str(int(np.argmax(digit_values)))
    if DEBUG and SHOW_JUDGE:
        cv2_show("Id Image", id_img)
    return student_id


def mix_picture(mask_image, non_mask_image, threshold=5):
    ret, mask = cv2.threshold(cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_BINARY)
    if DEBUG:
        cv2_show("Mask Image", mask_image)
        cv2_show("mask", mask)
    non_mask = cv2.bitwise_not(mask)
    foreground = cv2.bitwise_and(mask_image, mask_image, mask=mask)
    background = cv2.bitwise_and(non_mask_image, non_mask_image, mask=non_mask)
    image = cv2.add(foreground, background)
    return image


def process_frame(image, processed, gray_thresh, paper_area):
    student_id, score_value = None, None
    ret, thresh = cv2.threshold(processed, gray_thresh, 255, cv2.THRESH_BINARY)
    if DEBUG:
        cv2_show("Threshold", thresh)

    # 提取试卷轮廓并透视变换
    try:
        paper_contour = extract_contours_by_idx(thresh, 0)
        paper_pts = get_rectangle_corners(paper_contour)
        trans_img, trans_thresh = transform_regions(image, thresh, paper_pts)
        cv2_show('trans_img', trans_img)

        # 答案区域：查找矩形轮廓
        inv_thresh = cv2.bitwise_not(trans_thresh)
        contours = get_contours(inv_thresh)
        rect_contours = []
        for cnt in contours:
            if cv2.contourArea(cnt) > paper_area:
                arc = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * arc, True)
                if len(approx) == 4:
                    rect_contours.append(approx)
        if not rect_contours:
            raise ValueError("未检测到有效的答案区域。")
        answer_pts = get_rectangle_corners(rect_contours[0])
        answer_img, answer_thresh = transform_regions(trans_img, trans_thresh, answer_pts)
        answer_img = cv2.resize(answer_img, (737, 944))
        answer_thresh = cv2.resize(answer_thresh, (737, 944))
        cv2_show('answer_img', answer_img)

        score_value = process_questions(answer_img, answer_thresh)
        student_id = process_id(answer_img, answer_thresh)

        cv2.putText(answer_img, f"{score_value:.2f}%", (481, 841),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2_show('a', answer_img)

    except Exception as e:
        traceback.print_exc()

    return student_id, score_value


def main_progress(image_path_or_cap, file_path, gray_thresh=182, paper_area=500):
    grade_writer = GradeWriter(file_path)
    if isinstance(image_path_or_cap, str):
        image = cv2.imread(image_path_or_cap)
        image, processed = process_image(image)  # 读取图像及预处理
        student_id, score_value = process_frame(image, processed, gray_thresh, paper_area)
        grade_writer.append_grade(student_id, score_value)
        grade_writer.flush_grade_file()
    elif isinstance(image_path_or_cap, int):
        cap = cv2.VideoCapture(image_path_or_cap)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 获取图像尺寸
            (h, w) = frame.shape[:2]

            # 指定旋转角度（逆时针旋转45度）
            angle = 270

            # 计算图像中心
            center = (w // 2, h // 2)

            # 计算旋转矩阵
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # 进行仿射变换（旋转图像）
            frame = cv2.warpAffine(frame, M, (w, h))

            image, processed = process_image(frame)  # 读取图像及预处理
            student_id, score_value = process_frame(image, processed, gray_thresh, paper_area)
            clicked = cv2.waitKey(1) & 0xff
            if clicked == ord('q'):
                grade_writer.flush_grade_file()
                break
            elif clicked == ord('s'):
                grade_writer.append_grade(student_id, score_value)
            elif clicked == ord('f'):
                grade_writer.flush_grade_file()


if __name__ == '__main__':
    main_progress('1.jfif','grade.csv', gray_thresh=182, paper_area=5000)
