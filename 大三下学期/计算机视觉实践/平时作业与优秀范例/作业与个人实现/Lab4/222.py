# 先导入相关的库，定义显示图像函数
import cv2
import numpy as np
import matplotlib.pyplot as plt


def cv_show(name, img):  # 窗口名字，图像
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plt_show(name, img, sub=111):  # 标题，图像，子窗口id
    plt.subplot(sub)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(name)
    plt.axis("off")
    if (sub - 100) // 10 == (sub - 100) % 10:
        plt.show()

'''读入图像文件，转成灰度图'''
img = cv2.imread("1.jfif")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
ret1, thresh1 = cv2.threshold(imgBlur, 180, 255, cv2.THRESH_BINARY)
# plt_show('Set Threshold', thresh1)
print(ret1)
'''对阈值分割后的图像求轮廓'''
contours, hier = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))   # 2
imgContours = img.copy()
for i in range(len(contours)):
    cv2.drawContours(imgContours, contours, i, (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)), 2)
# plt_show('Contours',imgContours)
'''获得并显示轮廓封装成函数'''
def findShowContours(imgCont, imgShow):
    contours, hier = cv2.findContours(imgCont, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print('轮廓数量：', len(contours))

    imgContours = imgShow.copy()
    for i in range(len(contours)):
        cv2.drawContours(imgContours, contours, i, (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)), 2)
    plt_show('Contours',imgContours)
    return contours

# contours = findShowContours(thresh1, img)

#面积最大的轮廓
cnt = contours[0]
maxArea = cv2.contourArea(cnt)
maxCntId = 0
for i, cnt in enumerate(contours[1:], 1):
    area = cv2.contourArea(cnt)
    if area>maxArea:
        maxArea = area
        maxCntId = i

contour = contours[maxCntId]
imgContours = img.copy()
cv2.drawContours(imgContours, [contour], 0, (0, 255, 0), 2)
# plt_show('MaxContour',imgContours)
# 获取近似的轮廓
arc = cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, 0.02 * arc, True)
print(approx, approx.shape)
# 4个顶点
approx = approx.reshape((4,2))
imgConerPoints = img.copy()
for i, pnt in enumerate(approx):
    cv2.circle(imgConerPoints, (pnt[0], pnt[1]), 15, (0, 0, 255), -1)
    cv2.putText(imgConerPoints, f'{i}', (pnt[0], pnt[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0), 3)
# plt_show('ConerPoints',imgConerPoints)
def orderPoints(pts):
    '''对4点进行排序，下标0、1、2、3分别是左上、右上、右下、左下'''
    rect = np.zeros((4, 2), dtype="float32")
    # 计算左上和右下
    s = pts.sum(axis=1) #每个点的x和y相加，左上是和最小的，右下是和最大的。
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算右上和左下
    diff = np.diff(pts, axis=1)#每个点的y-x，右上最小，左下最大。
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

#排好序之后的结果
pntsOrded = orderPoints(approx)
print(pntsOrded)
imgConerPoints = img.copy()
for i, pnt in enumerate(pntsOrded):
    cv2.circle(imgConerPoints, (int(pnt[0]), int(pnt[1])), 15, (0, 0, 255), -1)
    cv2.putText(imgConerPoints, f'{i}', (int(pnt[0]), int(pnt[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0), 3)
# plt_show('ConerPoints',imgConerPoints)
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
    print("变换矩阵：", M)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    plt_show('Warped', warped)
    return warped


paper = perspectiveTransform(pntsOrded, img)
paperThresh = perspectiveTransform(pntsOrded, thresh1)

paperThreshInv = cv2.bitwise_not(paperThresh)
#也可以用下面这行代码实现反
#retInv, paperThreshInv = cv2.threshold(paperThresh,50,255,cv2.THRESH_BINARY_INV)
# plt_show('paperThreshInv', paperThreshInv)

contours = findShowContours(paperThreshInv, paper)

#获得形状是长方形且面积足够大的轮廓
rectCon = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 80:
        arc = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * arc, True)
        if len(approx) == 4:
            rectCon.append(approx)

# 根据面积排序
rectCon.sort(key=cv2.contourArea, reverse=True)
#
# # 绘制第2、3、4个轮廓
# for i in range(1, 4):  # 索引1, 2, 3对应第2, 3, 4个轮廓
#     if i < len(rectCon):
#         cv2.drawContours(paper, [rectCon[i]], -1, (0, 255, 0), 2)  # 使用绿色绘制轮廓
#
# # 显示结果
# cv2.imshow("Selected Contours", paper)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

imgRects = paper.copy()
rect1 = np.array(rectCon[1]).reshape(-1,1,2)  #最大的轮廓
cv2.polylines(imgRects, [rect1], True, (0,0,255), 3)

rect2 = np.array(rectCon[2]).reshape(-1,1,2)   #第二大的轮廓
cv2.polylines(imgRects, [rect2], True, (0,255,0), 3)

rect3 = np.array(rectCon[3]).reshape(-1,1,2)   #第二大的轮廓
cv2.polylines(imgRects, [rect3], True, (0,255,0), 3)
plt_show('imgRects',imgRects)

rect1Orded = orderPoints(rect2.reshape((-1,2))) #对最大轮廓的四个顶点排序
choices = perspectiveTransform(rect1Orded, paperThreshInv)[3:-3,3:-3]  #做透视变换并去掉边缘
choices=cv2.resize(choices,[1060,425])
ret, choices = cv2.threshold(choices, 100, 255, cv2.THRESH_BINARY)
rect1Color = choices.copy()
rect1Color = cv2.cvtColor(rect1Color, cv2.COLOR_GRAY2BGR)

contours = findShowContours(choices, rect1Color)
'''得到所有选项圆圈'''
choiceCnts = []
# 遍历
for cnt in contours:
    # 包围盒
    x, y, w, h = cv2.boundingRect(cnt)
    # 根据实际情况指定标准
    if w>=20 and h>=20 and x >=280:
        choiceCnts.append([cnt, (x, y, w, h)])

print(len(choiceCnts))

# for cnt, (x, y, w, h) in choiceCnts:
#     print(x, y)

# 反转列表，从上往下
choiceCnts.reverse()
choiceCnts.sort(key=lambda c: c[1][0])
# 分组并排序

# 打印并标出面积超过1200的轮廓
for question in choiceCnts:
    # for cnt, (x, y, w, h) in question:
        area = cv2.contourArea(question[0])  # 计算轮廓面积
        # if area > 1200:  # 如果面积超过1200
        (x,y,w,h)=question[1]
        print(f"Contour with area {area} at position: ({x}, {y})")
        # 在图像上绘制轮廓
        cv2.drawContours(choices, [cnt], -1, (0, 255, 0), 2)  # 用绿色绘制轮廓
        # 绘制矩形框
        cv2.rectangle(choices, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 用红色绘制矩形框
# Contour with area 773.5 at position: (12, 10)
# Contour with area 661.0 at position: (55, 10)
# Contour with area 1271.5 at position: (359, 20)
# Contour with area 1237.0 at position: (874, 21)
# Contour with area 1209.5 at position: (681, 22)
# Contour with area 1246.5 at position: (616, 60)
# Contour with area 1193.5 at position: (553, 61)
# Contour with area 1226.0 at position: (810, 61)
# Contour with area 1252.0 at position: (425, 99)
# Contour with area 1226.0 at position: (291, 102)
# Contour with area 1206.0 at position: (997, 220)
# Contour with area 1214.0 at position: (742, 334)

'''插入！！！！！'''
studentID =''
for cnt, (x, y, w, h) in choiceCnts:
    # print(x, y, w, h)

    if 10 <= y <= 30:
        studentID += '0'
    elif 50 <= y <=  70:
        studentID += '1'
    elif 90 <= y <=  110:
        studentID += '2'
    elif 130 <= y <=  150:
        studentID += '3'
    elif 160  <= y <=  180:
        studentID += '4'
    elif 200  <= y <=  220:
        studentID += '5'
    elif 240  <= y <= 260:
        studentID += '6'
    elif 280 <= y <= 300:
        studentID += '7'
    elif 320  <= y <=  340:
        studentID += '8'
    elif 360  <= y <=  380:
        studentID += '9'
print(studentID)
# 显示结果
# cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
# cv2.imshow("Contours", choices)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
rect2Orded = orderPoints(rect1.reshape((-1,2))) #对最大轮廓的四个顶点排序
choices1 = perspectiveTransform(rect2Orded, paperThreshInv)[3:-3,3:-3]  #做透视变换并去掉边缘
choices1 = cv2.resize(choices1,[1089,1200])
ret, choices1 = cv2.threshold(choices1, 100, 255, cv2.THRESH_BINARY)
rect1Color1 = choices1.copy()
rect1Color1 = cv2.cvtColor(rect1Color1, cv2.COLOR_GRAY2BGR)
contours1 = findShowContours(choices1, rect1Color1)
'''得到所有选项圆圈'''
choiceCnts1 = []
# 遍历
for cnt in contours1:
    # 包围盒
    x, y, w, h = cv2.boundingRect(cnt)
    # 根据实际情况指定标准
    if w >= 40 and h>=30 and w<=100:
        choiceCnts1.append([cnt, (x, y, w, h)])

# print(len(choiceCnts1))

# for cnt, (x, y, w, h) in choiceCnts:
#     print(x, y)

# 反转列表，从上往下
choiceCnts1.reverse()

# 分组并排序
# 打印并标出面积超过1200的轮廓
for question in choiceCnts1:
    # for cnt, (x, y, w, h) in question:
        area = cv2.contourArea(question[0])  # 计算轮廓面积
        (x,y,w,h)=question[1]
        print(f"Contour with area {area} at position: ({x}, {y},{w},{h})")
        # 在图像上绘制轮廓
        cv2.drawContours(choices1, [cnt], -1, (0, 255, 0), 2)  # 用绿色绘制轮廓
        # 绘制矩形框
        cv2.rectangle(choices1, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 用红色绘制矩形框


# 定义x,y中心点
x_centers = [60, 130, 200, 270, 440, 510, 580, 650, 820, 890, 960, 1030]
y_centers = [40,90,140,190,240,320,370,420,470,520,590,640,690,740,790,860,910,960,1010,1060]

# 直接录入答案
# correct_answers = ("B B B A C  "
#                    " B B D B A  "
#                    "D B B A A  "
#                    "D B B A A "
#                    "C A B A D "
#                    "C D A A D "
#                    "A D A A A "
#                    "A D D B D "
#                    "B D D A B "
#                    "D A A B A")
correct_answers ="B B D B B B B D B A B A C A A D C C B A D B B A A A A A D D A A B D D D A D D A B A A D B D A A B A"
correct_answers = correct_answers.replace(" ", "").upper()



# 建立坐标与选项的映射
answer_mapping = {}
for row in range(20):
    for col in range(12):
        x, y = x_centers[col], y_centers[row]
        answer_mapping[(col, row)] = (x, y)

# 根据中心匹配轮廓
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

print(detected_answers)


# 批改
correct_count = 0

for i in range(50):
    correct_ans = correct_answers[i]
    detected_ans = detected_answers[i]
    row = i // 3 if i < 45 else 15 + (i - 45)
    col_offset = (i % 3) * 4 if i < 45 else 0

    for opt_idx, opt in enumerate('ABCD'):
        col = col_offset + opt_idx
        x, y = answer_mapping[(col, row)]
        rect_pt1 = (x - 35, y - 25)
        rect_pt2 = (x + 35, y + 25)

        if opt == detected_ans:
            if opt == correct_ans:
                cv2.rectangle(choices1, rect_pt1, rect_pt2, (0, 255, 0), 2)  # 正确绿色
                correct_count += 1
            else:
                cv2.rectangle(choices1, rect_pt1, rect_pt2, (0, 0, 255), 2)  # 错误红色

        if opt == correct_ans and detected_ans != correct_ans:
            cv2.rectangle(choices1, rect_pt1, rect_pt2, (0, 255, 0), 2)  # 标记正确答案
        if detected_ans != correct_ans:
            print(f"第{i + 1}题错误，正确答案：{correct_ans}，你的答案：{detected_ans}")

# 计算并显示正确率
accuracy = correct_count / 50
accuracy_text = f'{int(accuracy * 100)}%'
cv2.putText(choices1, accuracy_text, (830, 1000), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 4)

choices_rgb = cv2.cvtColor(choices1, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,8))
plt.imshow(choices_rgb)
plt.axis('off')  # 不显示坐标轴
plt.title(f'Accuracy: {accuracy_text}')
plt.show()
