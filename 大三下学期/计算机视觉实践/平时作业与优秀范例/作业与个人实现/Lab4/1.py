import cv2
import numpy as np
import matplotlib.pyplot as plt

####################################
# 工具函数
####################################
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plt_show(name, img, sub=111):
    plt.subplot(sub)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(name)
    plt.axis("off")
    if (sub - 100) // 10 == (sub - 100) % 10:
        plt.show()

def findShowContours(imgCont, imgShow):
    contours, hier = cv2.findContours(imgCont, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print('轮廓数量：', len(contours))

    imgContours = imgShow.copy()
    for i in range(len(contours)):
        cv2.drawContours(imgContours, contours, i,
                         (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), 2)
    plt_show('Contours', imgContours)
    return contours

def orderPoints(pts):
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

def perspectiveTransform(ptsOrded, img):
    '''
    返回 透视变换后的图, 以及变换矩阵M
    '''
    (tl, tr, br, bl) = ptsOrded
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

    M = cv2.getPerspectiveTransform(ptsOrded, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped, M

def warpToOriginal(u, v, M_inv, offset_x, offset_y, scale_x, scale_y):
    # (u,v) 是在最终 choices1 坐标系中的像素坐标
    # 先反向缩放回 warped2_cropped
    u_cropped = u / scale_x + offset_x
    v_cropped = v / scale_y + offset_y
    # 现在在 warped2 坐标系

    src = np.array([[[u_cropped, v_cropped]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, M_inv)
    return int(dst[0][0][0]), int(dst[0][0][1])

####################################
# 1) 读取与预处理
####################################
img = cv2.imread("1.jfif")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
ret1, thresh1 = cv2.threshold(imgBlur, 180, 255, cv2.THRESH_BINARY)
print("阈值:", ret1)

contours, hier = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 寻找面积最大的轮廓
maxArea = cv2.contourArea(contours[0])
maxCntId = 0
for i, cnt in enumerate(contours[1:], 1):
    area = cv2.contourArea(cnt)
    if area > maxArea:
        maxArea = area
        maxCntId = i

contour = contours[maxCntId]
arc = cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, 0.02 * arc, True)
approx = approx.reshape((4, 2))

####################################
# 2) 获得并矫正纸张
####################################
pntsOrded = orderPoints(approx)
paper, M_paper = perspectiveTransform(pntsOrded, img)
paperThresh, M_thresh = perspectiveTransform(pntsOrded, thresh1)

paperThreshInv = cv2.bitwise_not(paperThresh)

# 寻找并排序长方形轮廓
contours = findShowContours(paperThreshInv, paper)
rectCon = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 80:
        arc = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * arc, True)
        if len(approx) == 4:
            rectCon.append(approx)

rectCon.sort(key=cv2.contourArea, reverse=True)

#############################
# (1) 标记rect2做warp
#############################
imgRects = paper.copy()
rect1 = np.array(rectCon[1]).reshape(-1,1,2)  # 最大轮廓（可自行改索引）
cv2.polylines(imgRects, [rect1], True, (0,0,255), 3)
rect2 = np.array(rectCon[2]).reshape(-1,1,2)
cv2.polylines(imgRects, [rect2], True, (0,255,0), 3)
rect3 = np.array(rectCon[3]).reshape(-1,1,2)
cv2.polylines(imgRects, [rect3], True, (0,255,0), 3)
plt_show('imgRects',imgRects)

####################################
# 3) 识别学号(示例在rect2)
####################################
rect1Orded = orderPoints(rect2.reshape((-1,2)))
warpedRect2, M_rect2 = perspectiveTransform(rect1Orded, paperThreshInv)

choices = warpedRect2[3:-3, 3:-3]
choices = cv2.resize(choices, [1060,425])
ret, choices = cv2.threshold(choices, 100, 255, cv2.THRESH_BINARY)
rect1Color = cv2.cvtColor(choices, cv2.COLOR_GRAY2BGR)

contours = findShowContours(choices, rect1Color)
choiceCnts = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w >= 20 and h >= 20 and x >= 280:
        choiceCnts.append([cnt, (x, y, w, h)])

print(len(choiceCnts))

# 反转列表，从上往下
choiceCnts.reverse()
choiceCnts.sort(key=lambda c: c[1][0])

studentID = ''
for cnt, (x, y, w, h) in choiceCnts:
    if 10 <= y <= 30:
        studentID += '0'
    elif 50 <= y <= 70:
        studentID += '1'
    elif 90 <= y <= 110:
        studentID += '2'
    elif 130 <= y <= 150:
        studentID += '3'
    elif 160 <= y <= 180:
        studentID += '4'
    elif 200 <= y <= 220:
        studentID += '5'
    elif 240 <= y <= 260:
        studentID += '6'
    elif 280 <= y <= 300:
        studentID += '7'
    elif 320 <= y <= 340:
        studentID += '8'
    elif 360 <= y <= 380:
        studentID += '9'
print("学号：", studentID)

####################################
# 4) 识别答题(示例在rect3)
####################################
rect2Orded = orderPoints(rect1.reshape((-1,2)))
warpedRect3, M_rect3 = perspectiveTransform(rect2Orded, paperThreshInv)

choices1_cropped = warpedRect3[3:-3, 3:-3]
choices1_cropped = cv2.resize(choices1_cropped, [1089,1200])
ret, choices1_cropped = cv2.threshold(choices1_cropped, 100, 255, cv2.THRESH_BINARY)
rect1Color1 = cv2.cvtColor(choices1_cropped, cv2.COLOR_GRAY2BGR)

contours1 = findShowContours(choices1_cropped, rect1Color1)
choiceCnts1 = []
for cnt in contours1:
    x, y, w, h = cv2.boundingRect(cnt)
    if w >= 40 and h >= 30 and w <= 100:
        choiceCnts1.append([cnt, (x, y, w, h)])

choiceCnts1.reverse()

for question in choiceCnts1:
    area = cv2.contourArea(question[0])
    (x, y, w, h) = question[1]
    print(f"Contour with area {area} at position: ({x}, {y},{w},{h})")

####################################
# 5) 答案批改
####################################
# 给出中心坐标
x_centers = [60, 130, 200, 270, 440, 510, 580, 650, 820, 890, 960, 1030]
y_centers = [40, 90, 140, 190, 240, 320, 370, 420, 470, 520, 590, 640, 690, 740, 790, 860, 910, 960, 1010, 1060]

# 录入答案
correct_answers = "B B D B B B B D B A B A C A A D C C B A D B B A A A A A D D A A B D D D A D D A B A A D B D A A B A"
correct_answers = correct_answers.replace(" ", "").upper()

# 建立坐标映射
answer_mapping = {}
for row in range(20):
    for col in range(12):
        x, y = x_centers[col], y_centers[row]
        answer_mapping[(col, row)] = (x, y)

# 在choices1_cropped空间查找，得到detected_answers
detected_answers = [''] * 50

for cnt, (x, y, w, h) in choiceCnts1:
    for row in range(20):
        if abs(y - y_centers[row]) <= 20:
            for col in range(12):
                if abs(x - x_centers[col]) <= 40:
                    q_idx = row * 3 + col // 4 if row < 15 else 45 + row - 15
                    option_idx = col % 4
                    option = 'ABCD'[option_idx]
                    detected_answers[q_idx] = option

# 批改统计
correct_count = 0

# 为了把绘制转回原图，要组合逆变换
H_warped3, W_warped3 = warpedRect3.shape[:2]
H_crop, W_crop = (H_warped3 - 6, W_warped3 - 6)  # 3:-3

# 但我们还对 cropped 做了 resize => (1089,1200)
final_w, final_h = 1089, 1200
scale_x = final_w / (W_warped3 - 6)
scale_y = final_h / (H_warped3 - 6)
M_rect3_inv = np.linalg.inv(M_rect3)

for i in range(50):
    correct_ans = correct_answers[i]
    detected_ans = detected_answers[i]

    row = i // 3 if i < 45 else 15 + (i - 45)
    col_offset = (i % 3) * 4 if i < 45 else 0

    for opt_idx, opt in enumerate('ABCD'):
        col = col_offset + opt_idx
        # 在最终warp坐标中的rect
        x_center, y_center = answer_mapping[(col, row)]
        rect_pt1 = (x_center - 35, y_center - 25)
        rect_pt2 = (x_center + 35, y_center + 25)

        # 判断对错
        if opt == detected_ans:
            if opt == correct_ans:
                color = (0, 255, 0)  # green
                correct_count += 1
            else:
                color = (0, 0, 255)  # red
        else:
            color = None

        # 若是正确答案但没选上，也标记
        if opt == correct_ans and detected_ans != correct_ans:
            color = (0, 255, 0)

        if color is not None:
            # 将rect_pt1, rect_pt2逆变换回原图坐标
            pt1_orig = warpToOriginal(rect_pt1[0], rect_pt1[1], M_rect3_inv, 18, 18, scale_x, scale_y)
            pt2_orig = warpToOriginal(rect_pt2[0], rect_pt2[1], M_rect3_inv, 18, 18, scale_x, scale_y)
            cv2.rectangle(img, pt1_orig, pt2_orig, color, 2)

    # 如果答错，则打印信息
    if detected_ans != correct_ans:
        print(f"第{i + 1}题错误，正确答案：{correct_ans}，你的答案：{detected_ans}")

####################################
# 6) 显示正确率在原图
####################################
accuracy = correct_count / 50
accuracy_text = f'{int(accuracy * 100)}%'
cv2.putText(img, accuracy_text, (900, 1650), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 3)

####################################
# 7) 显示最终结果(原图)
####################################
choices_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,8))
plt.imshow(choices_rgb)
plt.axis('off')  # 不显示坐标轴
plt.title(f'Accuracy: {accuracy_text}')
plt.savefig("final_result.png", bbox_inches='tight', dpi=300)  # 保存图像
plt.show()

# cv2.namedWindow("Final", cv2.WINDOW_NORMAL)
# cv2.imshow("Final", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
