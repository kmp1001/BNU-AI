import cv2
import numpy as np

answersGiven = [[0], [1], [2, 4], [0, 2], [3]]


def orderPoints(pts):  # 对4点进行排序
    # 共4个坐标点
    rect = np.zeros((4, 2), dtype="float32")
    # 按顺序找到对应坐标0、1、2、3分别是左上、右上、右下、左下
    # 计算左上和右下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算右上和左下
    diff = np.diff(pts, axis=1)
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


# 获得轮廓并随机颜色显示每条轮廓
def findShowContours(imgCont, imgShow):
    contours, hier = cv2.findContours(imgCont, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('轮廓数量：', len(contours))

    imgContours = imgShow.copy()
    for i in range(len(contours)):
        cv2.drawContours(imgContours, contours, i,
                         (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), 2)
    cv_show('Contours', imgContours)
    return contours


img = cv2.imread("Images/1.jpg")
cv2.imshow('Original Image', img)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)

# 阈值，得到答题卡区域
ret1, thresh1 = cv2.threshold(imgBlur, 180, 255, cv2.THRESH_BINARY)
cv2.imshow('thresh1', thresh1)

contours = findShowContours(thresh1, img)

# 最大轮廓是答题卡纸的轮廓
cnt = contours[0]
maxArea = cv2.contourArea(cnt)
maxCntId = 0
for i, cnt in enumerate(contours[1:], 1):
    area = cv2.contourArea(cnt)
    if area > maxArea:
        maxArea = area
        maxCntId = i

arc = cv2.arcLength(contours[maxCntId], True)
approx = cv2.approxPolyDP(contours[maxCntId], 0.1 * arc, True)
imgContours = img.copy()
cv2.drawContours(imgContours, [approx], -1, (0, 0, 255), 2)
cv2.imshow('Contours', imgContours)

# 透视变换
pntsOrded = orderPoints(approx.reshape((4, 2)))
paper = perspectiveTransform(pntsOrded, img)
paperThresh = perspectiveTransform(pntsOrded, thresh1)

paperThreshInv = cv2.bitwise_not(paperThresh)
cv2.imshow('paperThreshInv', paperThreshInv)

contours = findShowContours(paperThreshInv, paper)

# 筛选出两个正方形轮廓，一个对应答题区，一个对应分数区
rectCon = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 80:
        arc = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * arc, True)
        if len(approx) == 4:
            rectCon.append(approx)

rectCon.sort(key=cv2.contourArea, reverse=True)

rect1 = np.array(rectCon[0]).reshape(-1, 1, 2)  # 答题区
rect2 = np.array(rectCon[1]).reshape(-1, 1, 2)  # 分数区

# 对答题区透视变换
rect1Orded = orderPoints(rect1.reshape((-1, 2)))
choices = perspectiveTransform(rect1Orded, paperThreshInv)[3:-3, 3:-3]
ret, choices = cv2.threshold(choices, 100, 255, cv2.THRESH_BINARY)

rect1Contours = choices.copy()
rect1Contours = cv2.cvtColor(rect1Contours, cv2.COLOR_GRAY2BGR)
contours = findShowContours(choices, rect1Contours)

choiceCnts = []
# 遍历
for cnt in contours:
    # 包围盒
    x, y, w, h = cv2.boundingRect(cnt)
    # 根据实际情况指定标准，得到答题区的每一个圆形轮廓，即选项
    if w >= 30 and h >= 30:
        choiceCnts.append([cnt, (x, y, w, h)])

choiceCnts.reverse()
# 排序，按顺序得到每一道题的选项
questions = []
for i in range(len(choiceCnts) // 5):
    question = choiceCnts[i * 5:i * 5 + 5]
    question.sort(key=lambda b: b[1][0])
    questions.append(question)

print(choices.shape)
imgBlank1 = np.zeros((choices.shape[0], choices.shape[1], 3), dtype='uint8')
answers = []
score = 0
for j, one in enumerate(questions):
    oneAnsw = []  # 当前题目的答题情况
    for i, [cnt, (x, y, w, h)] in enumerate(one):
        imgChoice = choices[y:y + h, x:x + w]
        nonZeroNum = cv2.countNonZero(imgChoice)
        if nonZeroNum / (w * h) >= 0.4:  # 填涂面积比例大于0.5，则认为是选中该项
            oneAnsw.append(i)
            if i in answersGiven[j]:  # 选对，绿色
                cv2.circle(imgBlank1, (x + w // 2, y + h // 2), w // 2, (0, 255, 0), -1)
            else:  # 选错，红色
                cv2.circle(imgBlank1, (x + w // 2, y + h // 2), w // 2, (0, 0, 255), -1)
        else:  # 未选，黄色
            if i in answersGiven[j]:
                cv2.circle(imgBlank1, (x + w // 2, y + h // 2), w // 2, (0, 255, 255), -1)

    answers.append(oneAnsw)
    if answersGiven[j] == oneAnsw:  # 整道题正确则加分
        print(j)
        score += 20
print(score)
print(answersGiven)
print(answers)

# 逆变换回去显示答题情况
cv2.imshow('imgBlank1', imgBlank1)
invChoice = invPerspectiveTransform(rect1Orded, imgBlank1, paper)
invChoiceFinal = invPerspectiveTransform(pntsOrded, invChoice, img)

imgFinal = img.copy()
imgFinal = cv2.add(imgFinal, invChoiceFinal)

# 分数逆变换回去并显示
rect2Orded = orderPoints(rect2.reshape((-1, 2)))
warpedGrade = perspectiveTransform(rect2Orded, paper)

imgBlank2 = np.zeros_like(warpedGrade)
cv2.putText(imgBlank2, str(int(score)) + "%", (5, 35)
            , cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

invGrade = invPerspectiveTransform(rect2Orded, imgBlank2, paper)
invGradeFinal = invPerspectiveTransform(pntsOrded, invGrade, img)

invGradeFinalGray = cv2.cvtColor(invGradeFinal, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(invGradeFinalGray, 50, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

background = cv2.bitwise_and(imgFinal, imgFinal, mask=mask_inv)
foreground = cv2.bitwise_and(invGradeFinal, invGradeFinal, mask=mask)
imgFinal = cv2.add(foreground, background)
cv2.imshow('imgFinal', imgFinal)

