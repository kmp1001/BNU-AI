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

# 读取图像
img = cv2.imread('Images/harrypotter.png')
if img is None:
    print("找不到图像文件，请检查路径！")
    exit()
img2 = img.copy()  # 保存原始图像
result = np.zeros(img.shape, np.uint8)  # 分割结果图像初始化为全黑

# 全局变量初始化
drawing = False         # 标记绘制（scribble）状态，用于前景/背景标记
lbd = False             # 标记鼠标左键状态（用于矩形绘制）
mode = 'rect'           # 当前模式：'rect' - 绘制矩形；'fgd' - 标记前景；'bgd' - 标记背景
rect = None             # 矩形区域 (x, y, w, h)
brush_radius = 3        # 绘制标记时的画笔半径

# GrabCut 参数初始化
mask = np.zeros(img.shape[:2], dtype=np.uint8)  # GrabCut mask，所有像素初始值为0
bgdModel = np.zeros((1, 65), np.float64)          # 背景模型
fgdModel = np.zeros((1, 65), np.float64)          # 前景模型

# 鼠标回调函数，根据不同模式执行不同操作
def mouseEvents(event, x, y, flags, param):
    global img, img2, drawing, lbd, startX, startY, rect, mode, mask, brush_radius

    if mode == 'rect':  # 矩形选择模式
        if event == cv2.EVENT_LBUTTONDOWN:
            startX, startY = x, y
            lbd = True
        elif event == cv2.EVENT_MOUSEMOVE and lbd:
            img = img2.copy()
            cv2.rectangle(img, (startX, startY), (x, y), (255, 0, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            rect = (min(startX, x), min(startY, y), abs(x - startX), abs(y - startY))
            lbd = False
            # 绘制最终矩形
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 3)
            print("已选定矩形区域:", rect)
    elif mode in ['fgd', 'bgd']:  # 前景或背景标记模式
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            if mode == 'fgd':
                cv2.circle(img, (x, y), brush_radius, (0, 255, 0), -1)  # 前景用绿色
                cv2.circle(mask, (x, y), brush_radius, cv2.GC_FGD, -1)
            else:  # 'bgd'
                cv2.circle(img, (x, y), brush_radius, (0, 0, 255), -1)  # 背景用红色
                cv2.circle(mask, (x, y), brush_radius, cv2.GC_BGD, -1)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            if mode == 'fgd':
                cv2.circle(img, (x, y), brush_radius, (0, 255, 0), -1)
                cv2.circle(mask, (x, y), brush_radius, cv2.GC_FGD, -1)
            else:
                cv2.circle(img, (x, y), brush_radius, (0, 0, 255), -1)
                cv2.circle(mask, (x, y), brush_radius, cv2.GC_BGD, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

# 设置窗口和鼠标回调
cv2.namedWindow('GrabCut')
cv2.setMouseCallback('GrabCut', mouseEvents)

print("使用说明：")
print("1. 首先在窗口中拖动鼠标选择前景区域的矩形 (矩形模式)。")
print("2. 按键 '0' 切换到背景标记模式，用鼠标绘制背景标记。")
print("3. 按键 '1' 切换到前景标记模式，用鼠标绘制前景标记。")
print("4. 按 'c' 键运行GrabCut算法进行分割。")
print("5. 按 'r' 键重置图像和所有标记。")
print("6. 按 'q' 键退出程序。")

while True:
    cv2.imshow('GrabCut', img)
    cv2.imshow('Result', result)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break
    elif k == ord('r'):  # 重置所有内容
        print("重置所有设置。")
        mode = 'rect'
        rect = None
        drawing = False
        lbd = False
        img = img2.copy()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        result = np.zeros(img.shape, np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
    elif k == ord('0'):  # 切换到背景标记模式
        mode = 'bgd'
        print("当前模式：背景标记")
    elif k == ord('1'):  # 切换到前景标记模式
        mode = 'fgd'
        print("当前模式：前景标记")
    elif k == ord('c'):
        if rect is None:
            print("请先用鼠标选择前景的矩形区域！")
            continue
        # 第一次运行 GrabCut 时，如果 mask 还没有被更新，则使用矩形初始化
        if np.count_nonzero(mask) == 0:
            cv2.grabCut(img2, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
        else:
            cv2.grabCut(img2, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
        # 根据 mask 提取前景
        mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        result = cv2.bitwise_and(img2, img2, mask=mask2)
        print("GrabCut 算法运行完成。")

cv2.destroyAllWindows()
