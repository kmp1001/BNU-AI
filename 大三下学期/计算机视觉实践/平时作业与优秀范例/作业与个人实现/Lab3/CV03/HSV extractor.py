import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# 创建隐藏的Tkinter窗口，用于弹出文件选择对话框
root = tk.Tk()
root.withdraw()

# 弹出文件选择对话框，选择一张图片
file_path = filedialog.askopenfilename(title="请选择图片",
                                       filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
if not file_path:
    print("未选择图片，程序退出。")
    exit()

# 读取图片
image = cv2.imread(file_path)
if image is None:
    print("无法加载图片，请检查文件路径或图片格式。")
    exit()

# 定义鼠标点击回调函数
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 获取点击处的BGR颜色值
        bgr = image[y, x]
        # 将BGR颜色转换为HSV颜色
        pixel = np.uint8([[bgr]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
        # 输出点击坐标、BGR值和HSV值
        print(f"点击位置: ({x}, {y})")
        print(f"BGR值: {bgr}")
        print(f"HSV值: {hsv[0][0]}")

# 创建窗口并设置鼠标回调函数
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

# 显示图片，并等待用户按 'q' 键退出
while True:
    cv2.imshow('Image', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
#
# import cv2
# import numpy as np
#
# # 1. 读取图像
# # 建议使用绝对路径或将图像放在同级目录
# img = cv2.imread('Puzzle9.png')
# # 如果读取失败，请检查路径或文件名
# if img is None:
#     raise FileNotFoundError("图像读取失败，请检查文件路径。")
#
# # 2. 转为灰度图
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # 3. 边缘检测（Canny）
# #   - 这里的阈值(100, 200)需要根据实际图像情况微调
# edges = cv2.Canny(gray, 100, 200)
#
# # 4. 形态学操作
# #   - MORPH_CLOSE可以将边缘之间的小缝隙闭合，从而让外部轮廓变得更连续
# #   - kernel 大小同样需要根据线条粗细和图像分辨率进行调参
# kernel = np.ones((5, 5), np.uint8)
# closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
#
# # 5. 查找轮廓（只找最外层轮廓）
# #    - 这里使用 RETR_EXTERNAL 只保留最外层轮廓，如果需要保留内孔，可以用 RETR_TREE 或 RETR_CCOMP
# contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # 6. 可视化轮廓
# #    - 在原图上绘制找到的所有外部轮廓
# output = img.copy()
# cv2.drawContours(output, contours, -1, (0, 0, 255), 2)
#
# # 7. 保存结果
# cv2.imwrite('maze_external_contours.png', output)
#
# print("处理完成，结果已保存为 maze_external_contours.png。")
