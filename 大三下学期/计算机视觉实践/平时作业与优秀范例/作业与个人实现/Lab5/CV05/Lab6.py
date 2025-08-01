# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def cv_show(name, img):
#     cv2.imshow(name, img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# def plt_show(name, img, sub=111):
#     plt.subplot(sub)
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title(name)
#     plt.axis("off")
#     if (sub - 100) // 10 == (sub - 100) % 10:
#         plt.show()
#
#
# # 读取图像
# img = cv2.imread('Images/Count/chuanchuan.png')
# if img is None:
#     print("找不到图像文件，请检查路径！")
#     exit()
# img2 = img.copy()  # 保存原始图像
# result = np.zeros(img.shape, np.uint8)  # 分割结果图像初始化为全黑
#
# # 全局变量初始化
# drawing = False  # 标记绘制（scribble）状态，用于前景/背景标记
# lbd = False  # 标记鼠标左键状态（用于矩形绘制）
# mode = 'rect'  # 当前模式：'rect' - 绘制矩形；'fgd' - 标记前景；'bgd' - 标记背景
# rect = None  # 矩形区域 (x, y, w, h)
# brush_radius = 3  # 绘制标记时的画笔半径
#
# # GrabCut 参数初始化
# mask = np.zeros(img.shape[:2], dtype=np.uint8)  # GrabCut mask，所有像素初始值为0
# bgdModel = np.zeros((1, 65), np.float64)  # 背景模型
# fgdModel = np.zeros((1, 65), np.float64)  # 前景模型
#
#
# # 鼠标回调函数，根据不同模式执行不同操作
# def mouseEvents(event, x, y, flags, param):
#     global img, img2, drawing, lbd, startX, startY, rect, mode, mask, brush_radius
#
#     if mode == 'rect':  # 矩形选择模式
#         if event == cv2.EVENT_LBUTTONDOWN:
#             startX, startY = x, y
#             lbd = True
#         elif event == cv2.EVENT_MOUSEMOVE and lbd:
#             img = img2.copy()
#             cv2.rectangle(img, (startX, startY), (x, y), (255, 0, 0), 2)
#         elif event == cv2.EVENT_LBUTTONUP:
#             rect = (min(startX, x), min(startY, y), abs(x - startX), abs(y - startY))
#             lbd = False
#             cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 3)
#             print("已选定矩形区域:", rect)
#     elif mode in ['fgd', 'bgd']:  # 前景或背景标记模式
#         if event == cv2.EVENT_LBUTTONDOWN:
#             drawing = True
#             if mode == 'fgd':
#                 cv2.circle(img, (x, y), brush_radius, (0, 255, 0), -1)  # 前景用绿色
#                 cv2.circle(mask, (x, y), brush_radius, cv2.GC_FGD, -1)
#             else:  # 'bgd'
#                 cv2.circle(img, (x, y), brush_radius, (0, 0, 255), -1)  # 背景用红色
#                 cv2.circle(mask, (x, y), brush_radius, cv2.GC_BGD, -1)
#         elif event == cv2.EVENT_MOUSEMOVE and drawing:
#             if mode == 'fgd':
#                 cv2.circle(img, (x, y), brush_radius, (0, 255, 0), -1)
#                 cv2.circle(mask, (x, y), brush_radius, cv2.GC_FGD, -1)
#             else:
#                 cv2.circle(img, (x, y), brush_radius, (0, 0, 255), -1)
#                 cv2.circle(mask, (x, y), brush_radius, cv2.GC_BGD, -1)
#         elif event == cv2.EVENT_LBUTTONUP:
#             drawing = False
#
#
# # 设置窗口和鼠标回调
# cv2.namedWindow('GrabCut')
# cv2.setMouseCallback('GrabCut', mouseEvents)
#
# print("使用说明：")
# print("1. 首先在窗口中拖动鼠标选择前景区域的矩形 (矩形模式)。")
# print("2. 按键 '0' 切换到背景标记模式，用鼠标绘制背景标记。")
# print("3. 按键 '1' 切换到前景标记模式，用鼠标绘制前景标记。")
# print("4. 按 'c' 键运行GrabCut算法进行分割。")
# print("5. 按 'r' 键重置图像和所有标记。")
# print("6. 按 'q' 键退出程序。")
#
# while True:
#     cv2.imshow('GrabCut', img)
#     cv2.imshow('Result', result)
#     k = cv2.waitKey(1) & 0xFF
#
#     if k == ord('q'):
#         break
#     elif k == ord('r'):  # 重置所有内容
#         print("重置所有设置。")
#         mode = 'rect'
#         rect = None
#         drawing = False
#         lbd = False
#         img = img2.copy()
#         mask = np.zeros(img.shape[:2], dtype=np.uint8)
#         result = np.zeros(img.shape, np.uint8)
#         bgdModel = np.zeros((1, 65), np.float64)
#         fgdModel = np.zeros((1, 65), np.float64)
#     elif k == ord('0'):  # 切换到背景标记模式
#         mode = 'bgd'
#         print("当前模式：背景标记")
#     elif k == ord('1'):  # 切换到前景标记模式
#         mode = 'fgd'
#         print("当前模式：前景标记")
#     elif k == ord('c'):
#         if rect is None:
#             print("请先用鼠标选择前景的矩形区域！")
#             continue
#         # 第一次运行 GrabCut 时，如果 mask 还没有被更新，则使用矩形初始化
#         if np.count_nonzero(mask) == 0:
#             cv2.grabCut(img2, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
#         else:
#             cv2.grabCut(img2, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
#         # 根据 mask 提取前景
#         mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
#         result = cv2.bitwise_and(img2, img2, mask=mask2)
#
#         # ----- 下面开始对 GrabCut 得到的前景区域中的小圆形进行检测 -----
#         # 转换为灰度图，并进行模糊处理（根据需要选择合适的模糊方法）
#         gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
#         # gray_blurred = cv2.medianBlur(gray, 1)
#         # 在模糊后加上腐蚀操作
#         # kernel = np.ones((3, 3), np.uint8)
#         # eroded = cv2.erode(gray_blurred, kernel, iterations=1)
#
#         # 使用霍夫圆检测
#         # 参数说明：dp=1 表示累加器分辨率与原图相同，minDist 表示检测到的圆之间最小距离，
#         # param1 和 param2 分别对应边缘检测的高阈值和圆心检测的累加器阈值，
#         # minRadius 和 maxRadius 分别为圆半径范围，根据图像情况调整
#         ## cells参数
#         # circles = cv2.HoughCircles(gray,
#         #                            cv2.HOUGH_GRADIENT,
#         #                            dp=1,
#         #                            minDist=5,
#         #                            param1=30,
#         #                            param2=20,
#         #                            minRadius=5,
#         #                            maxRadius=20)
#         ## chuanchuan参数
#         circles = cv2.HoughCircles(gray,
#                                    cv2.HOUGH_GRADIENT,
#                                    dp=1,
#                                    minDist=10,
#                                    param1=30,
#                                    param2=20,
#                                    minRadius=10,
#                                    maxRadius=20)
#
#         circle_count = 0
#         if circles is not None:
#             circles = np.uint16(np.around(circles))
#             for i in circles[0, :]:
#                 # 在分割结果上绘制圆形和圆心
#                 cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
#                 cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)
#                 circle_count += 1
#             print("检测到的小圆形数量：", circle_count)
#         else:
#             print("未检测到小圆形！")
#
#         # 显示检测结果
#         cv2.putText(result,f"{circle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow("Detected Circles", result)
#         cv2.imwrite("Detected_Circles_chuanchuan.png", result)
#
# cv2.destroyAllWindows()
#
#
# # chuanchuan:111
#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def cv_show(name, img):
#     cv2.namedWindow(name, cv2.WINDOW_NORMAL)  # 允许窗口大小调整
#     cv2.imshow(name, img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# def plt_show(name, img, sub=111):
#     plt.subplot(sub)
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title(name)
#     plt.axis("off")
#     if (sub - 100) // 10 == (sub - 100) % 10:
#         plt.show()
#
#
# # 读取图像
# img = cv2.imread('Images/Count/seeds2.png')
# if img is None:
#     print("找不到图像文件，请检查路径！")
#     exit()
# img2 = img.copy()  # 保存原始图像
# result = np.zeros(img.shape, np.uint8)  # 分割结果图像初始化为全黑
#
# # 全局变量初始化
# drawing = False  # 标记绘制（scribble）状态，用于前景/背景标记
# lbd = False      # 标记鼠标左键状态（用于矩形绘制）
# mode = 'rect'    # 当前模式：'rect' - 绘制矩形；'fgd' - 标记前景；'bgd' - 标记背景
# rect = None      # 矩形区域 (x, y, w, h)
# brush_radius = 3 # 绘制标记时的画笔半径
#
# # GrabCut 参数初始化
# mask = np.zeros(img.shape[:2], dtype=np.uint8)  # GrabCut mask，所有像素初始值为0
# bgdModel = np.zeros((1, 65), np.float64)          # 背景模型
# fgdModel = np.zeros((1, 65), np.float64)          # 前景模型
#
#
# # 鼠标回调函数，根据不同模式执行不同操作
# def mouseEvents(event, x, y, flags, param):
#     global img, img2, drawing, lbd, startX, startY, rect, mode, mask, brush_radius
#
#     if mode == 'rect':  # 矩形选择模式
#         if event == cv2.EVENT_LBUTTONDOWN:
#             startX, startY = x, y
#             lbd = True
#         elif event == cv2.EVENT_MOUSEMOVE and lbd:
#             img = img2.copy()
#             cv2.rectangle(img, (startX, startY), (x, y), (255, 0, 0), 2)
#         elif event == cv2.EVENT_LBUTTONUP:
#             rect = (min(startX, x), min(startY, y), abs(x - startX), abs(y - startY))
#             lbd = False
#             cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 3)
#             print("已选定矩形区域:", rect)
#     elif mode in ['fgd', 'bgd']:  # 前景或背景标记模式
#         if event == cv2.EVENT_LBUTTONDOWN:
#             drawing = True
#             if mode == 'fgd':
#                 cv2.circle(img, (x, y), brush_radius, (0, 255, 0), -1)  # 前景用绿色
#                 cv2.circle(mask, (x, y), brush_radius, cv2.GC_FGD, -1)
#             else:
#                 cv2.circle(img, (x, y), brush_radius, (0, 0, 255), -1)  # 背景用红色
#                 cv2.circle(mask, (x, y), brush_radius, cv2.GC_BGD, -1)
#         elif event == cv2.EVENT_MOUSEMOVE and drawing:
#             if mode == 'fgd':
#                 cv2.circle(img, (x, y), brush_radius, (0, 255, 0), -1)
#                 cv2.circle(mask, (x, y), brush_radius, cv2.GC_FGD, -1)
#             else:
#                 cv2.circle(img, (x, y), brush_radius, (0, 0, 255), -1)
#                 cv2.circle(mask, (x, y), brush_radius, cv2.GC_BGD, -1)
#         elif event == cv2.EVENT_LBUTTONUP:
#             drawing = False
#
# # 创建可调整大小的窗口
# cv2.namedWindow('GrabCut', cv2.WINDOW_NORMAL)
# cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
#
# cv2.setMouseCallback('GrabCut', mouseEvents)
#
# print("使用说明：")
# print("1. 首先在窗口中拖动鼠标选择前景区域的矩形 (矩形模式)。")
# print("2. 按键 '0' 切换到背景标记模式，用鼠标绘制背景标记。")
# print("3. 按键 '1' 切换到前景标记模式，用鼠标绘制前景标记。")
# print("4. 按 'c' 键运行GrabCut算法进行分割。")
# print("5. 按 'r' 键重置图像和所有标记。")
# print("6. 按 'q' 键退出程序。")
#
# while True:
#     cv2.imshow('GrabCut', img)
#     cv2.imshow('Result', result)
#     k = cv2.waitKey(1) & 0xFF
#
#     if k == ord('q'):
#         break
#     elif k == ord('r'):  # 重置所有内容
#         print("重置所有设置。")
#         mode = 'rect'
#         rect = None
#         drawing = False
#         lbd = False
#         img = img2.copy()
#         mask = np.zeros(img.shape[:2], dtype=np.uint8)
#         result = np.zeros(img.shape, np.uint8)
#         bgdModel = np.zeros((1, 65), np.float64)
#         fgdModel = np.zeros((1, 65), np.float64)
#     elif k == ord('0'):  # 切换到背景标记模式
#         mode = 'bgd'
#         print("当前模式：背景标记")
#     elif k == ord('1'):  # 切换到前景标记模式
#         mode = 'fgd'
#         print("当前模式：前景标记")
#     elif k == ord('c'):
#         if rect is None:
#             print("请先用鼠标选择前景的矩形区域！")
#             continue
#         # 第一次运行 GrabCut 时，如果 mask 还没有被更新，则使用矩形初始化
#         if np.count_nonzero(mask) == 0:
#             cv2.grabCut(img2, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
#         else:
#             cv2.grabCut(img2, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
#         # 根据 mask 提取前景
#         mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
#         result = cv2.bitwise_and(img2, img2, mask=mask2)
#
#         # ----- 下面开始对 GrabCut 得到的前景区域中的小圆形进行检测 -----
#         # 转换为灰度图
#         gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
#         # 使用霍夫圆检测 (seeds2参数)
#         circles = cv2.HoughCircles(gray,
#                                    cv2.HOUGH_GRADIENT,
#                                    dp=1,
#                                    minDist=10,
#                                    param1=30,
#                                    param2=20,
#                                    minRadius=12,
#                                    maxRadius=20)
#
#         # circles = cv2.HoughCircles(gray,
#         #                            cv2.HOUGH_GRADIENT,
#         #                            dp=1,
#         #                            minDist=15,
#         #                            param1=30,
#         #                            param2=20,
#         #                            minRadius=15,
#         #                            maxRadius=20)
#         circle_count = 0
#         if circles is not None:
#             circles = np.uint16(np.around(circles))
#             for i in circles[0, :]:
#                 # 在分割结果上绘制圆形和圆心
#                 cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
#                 cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)
#                 circle_count += 1
#             print("检测到的小圆形数量：", circle_count)
#         else:
#             print("未检测到小圆形！")
#
#         # 显示检测结果（同样使用可调整大小的窗口）
#
#         # cv2.imshow("Detected Circles", result)
#         # cv2.imwrite("Detected_Circles_seeds2.png", result)
#         cv2.namedWindow("Detected Circles", cv2.WINDOW_NORMAL)
#         cv2.putText(result, f"{circle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow("Detected Circles", result)
#         cv2.imwrite("Detected_Circles_seeds2.png", result)
#
# cv2.destroyAllWindows()

#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def cv_show(name, img):
#     cv2.namedWindow(name, cv2.WINDOW_NORMAL)  # 允许调整窗口大小
#     cv2.imshow(name, img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# def plt_show(name, img, sub=111):
#     plt.subplot(sub)
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title(name)
#     plt.axis("off")
#     if (sub - 100) // 10 == (sub - 100) % 10:
#         plt.show()
#
#
# # 读取图像
# img = cv2.imread('Images/Count/squares.png')
# if img is None:
#     print("找不到图像文件，请检查路径！")
#     exit()
# img2 = img.copy()  # 保存原始图像
# result = np.zeros(img.shape, np.uint8)  # 分割结果图像初始化为全黑
#
# # 全局变量初始化
# drawing = False  # 标记绘制（scribble）状态，用于前景/背景标记
# lbd = False  # 标记鼠标左键状态（用于矩形绘制）
# mode = 'rect'  # 当前模式：'rect' - 绘制矩形；'fgd' - 标记前景；'bgd' - 标记背景
# rect = None  # 矩形区域 (x, y, w, h)
# brush_radius = 3  # 绘制标记时的画笔半径
#
# # GrabCut 参数初始化
# mask = np.zeros(img.shape[:2], dtype=np.uint8)  # GrabCut mask，所有像素初始值为0
# bgdModel = np.zeros((1, 65), np.float64)  # 背景模型
# fgdModel = np.zeros((1, 65), np.float64)  # 前景模型
#
#
# # 鼠标回调函数，根据不同模式执行不同操作
# def mouseEvents(event, x, y, flags, param):
#     global img, img2, drawing, lbd, startX, startY, rect, mode, mask, brush_radius
#
#     if mode == 'rect':  # 矩形选择模式
#         if event == cv2.EVENT_LBUTTONDOWN:
#             startX, startY = x, y
#             lbd = True
#         elif event == cv2.EVENT_MOUSEMOVE and lbd:
#             img = img2.copy()
#             cv2.rectangle(img, (startX, startY), (x, y), (255, 0, 0), 2)
#         elif event == cv2.EVENT_LBUTTONUP:
#             rect = (min(startX, x), min(startY, y), abs(x - startX), abs(y - startY))
#             lbd = False
#             cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 3)
#             print("已选定矩形区域:", rect)
#     elif mode in ['fgd', 'bgd']:  # 前景或背景标记模式
#         if event == cv2.EVENT_LBUTTONDOWN:
#             drawing = True
#             if mode == 'fgd':
#                 cv2.circle(img, (x, y), brush_radius, (0, 255, 0), -1)  # 前景用绿色
#                 cv2.circle(mask, (x, y), brush_radius, cv2.GC_FGD, -1)
#             else:
#                 cv2.circle(img, (x, y), brush_radius, (0, 0, 255), -1)  # 背景用红色
#                 cv2.circle(mask, (x, y), brush_radius, cv2.GC_BGD, -1)
#         elif event == cv2.EVENT_MOUSEMOVE and drawing:
#             if mode == 'fgd':
#                 cv2.circle(img, (x, y), brush_radius, (0, 255, 0), -1)
#                 cv2.circle(mask, (x, y), brush_radius, cv2.GC_FGD, -1)
#             else:
#                 cv2.circle(img, (x, y), brush_radius, (0, 0, 255), -1)
#                 cv2.circle(mask, (x, y), brush_radius, cv2.GC_BGD, -1)
#         elif event == cv2.EVENT_LBUTTONUP:
#             drawing = False
#
#
# # 创建可调整大小的窗口
# cv2.namedWindow('GrabCut', cv2.WINDOW_NORMAL)
# cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
# cv2.setMouseCallback('GrabCut', mouseEvents)
#
# print("使用说明：")
# print("1. 首先在窗口中拖动鼠标选择前景区域的矩形 (矩形模式)。")
# print("2. 按键 '0' 切换到背景标记模式，用鼠标绘制背景标记。")
# print("3. 按键 '1' 切换到前景标记模式，用鼠标绘制前景标记。")
# print("4. 按 'c' 键运行 GrabCut 算法进行分割。")
# print("5. 按 'r' 键重置图像和所有标记。")
# print("6. 按 'q' 键退出程序。")
#
# while True:
#     cv2.imshow('GrabCut', img)
#     cv2.imshow('Result', result)
#     k = cv2.waitKey(1) & 0xFF
#
#     if k == ord('q'):
#         break
#     elif k == ord('r'):  # 重置所有内容
#         print("重置所有设置。")
#         mode = 'rect'
#         rect = None
#         drawing = False
#         lbd = False
#         img = img2.copy()
#         mask = np.zeros(img.shape[:2], dtype=np.uint8)
#         result = np.zeros(img.shape, np.uint8)
#         bgdModel = np.zeros((1, 65), np.float64)
#         fgdModel = np.zeros((1, 65), np.float64)
#     elif k == ord('0'):  # 切换到背景标记模式
#         mode = 'bgd'
#         print("当前模式：背景标记")
#     elif k == ord('1'):  # 切换到前景标记模式
#         mode = 'fgd'
#         print("当前模式：前景标记")
#     elif k == ord('c'):
#         if rect is None:
#             print("请先用鼠标选择前景的矩形区域！")
#             continue
#         # 第一次运行 GrabCut 时，如果 mask 还没有被更新，则使用矩形初始化
#         if np.count_nonzero(mask) == 0:
#             cv2.grabCut(img2, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
#         else:
#             cv2.grabCut(img2, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
#         # 根据 mask 提取前景
#         mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
#         result = cv2.bitwise_and(img2, img2, mask=mask2)
#
#         # ----- 下面开始对 GrabCut 得到的前景区域中的矩形/正方形目标进行检测 -----
#         # 将前景区域转换为灰度图
#         gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
#         # 使用 Otsu 阈值分割
#         ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         # cv2.imshow("Threshold", thresh)
#         # ret, thresh = cv2.threshold(gray, 0, 255, cv2.TH)
#         # 形态学操作：先开运算去噪，再闭运算填补空洞（参数可根据实际情况调整）
#         # kernel = np.ones((3, 3), np.uint8)
#         # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
#         # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
#
#         # 查找轮廓
#         contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#
#         rect_count = 0
#         detected_img = img2.copy()
#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             # if area < 10:  # 过滤面积过小的噪点，根据实际情况调整
#             #     continue
#             # 多边形近似
#             epsilon = 0.02 * cv2.arcLength(cnt, True)
#             approx = cv2.approxPolyDP(cnt, epsilon, True)
#             # 判断是否为四边形且凸，认为是近似矩形或正方形
#             if len(approx) == 4 and cv2.isContourConvex(approx):
#             # if cv2.isContourConvex(approx):
#                 # 计算外接矩形
#                 x, y, w, h = cv2.boundingRect(approx)
#                 aspect_ratio = float(w) / h
#                 # 可根据需要筛选出接近正方形（例如 aspect_ratio 接近 1）或者较宽松的矩形
#                 if 0.25 < aspect_ratio < 4.0:
#                     rect_count += 1
#                     cv2.drawContours(detected_img, [approx], 0, (0, 255, 0), 2)
#                     # 绘制中心点
#                     cx = x + w // 2
#                     cy = y + h // 2
#                     cv2.circle(detected_img, (cx, cy), 2, (0, 0, 255), -1)
#
#         print("检测到的矩形/正方形数量：", rect_count)
#         cv2.namedWindow("Detected Rectangles", cv2.WINDOW_NORMAL)
#         cv2.imshow("Detected Rectangles", detected_img)
#
# cv2.destroyAllWindows()
#

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# def plt_show(name, img, sub=111):
#     plt.subplot(sub)
#     # 将 BGR 转为 RGB 后显示
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title(name)
#     plt.axis("off")
#     # 当绘制完一行时显示图像
#     if (sub - 100) // 10 == (sub - 100) % 10:
#         plt.show()
#
# # 读取图像（这里用的是小方块图像，可根据需要修改路径）
# img = cv2.imread('Images/Count/seeds1.png')
# if img is None:
#     print("找不到图像文件，请检查路径！")
#     exit()
# img2 = img.copy()  # 保存原始图像
#
#
# # 将图像转换为灰度图，并进行高斯模糊
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
#
# plt.figure(1, figsize=(12,10))
# plt_show('Orig img', img, 131)
#
# # 全局阈值分割（二值化），阈值参数可根据图像调整
# ret, thresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
# plt_show("Threshold", thresh, 132)
#
# # 自适应阈值分割，去噪后得到边缘信息
# thresh2 = cv2.adaptiveThreshold(imgBlur, 255,
#                                 cv2.ADAPTIVE_THRESH_MEAN_C,
#                                 cv2.THRESH_BINARY_INV, 9, 8)
# thresh2 = cv2.medianBlur(thresh2, 3)
# plt_show('Adaptive Threshold', thresh2, 133)
#
# # 膨胀操作，使边缘更连通
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# dilate = cv2.morphologyEx(thresh2, cv2.MORPH_DILATE, kernel, iterations=1)
# plt.figure(2, figsize=(12,10))
# plt_show("Dilate", dilate, 131)
#
# # 差分：利用全局阈值图像减去膨胀后的自适应阈值图像，分离相邻块
# diff = cv2.subtract(thresh, dilate)
# plt_show("Diff", diff, 132)
#
# # 腐蚀操作，进一步去除细小连接
# erode = cv2.erode(diff, kernel, iterations=2)
# plt_show("Erode", erode, 133)
#
# # 提取轮廓，使用 RETR_EXTERNAL 只关注外部轮廓
# contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # 遍历轮廓，对面积大于一定阈值的块求最小内接圆，并在原图上填充随机颜色
# count = 0
# imgContours = img.copy()
# for cnt in contours:
#     area = cv2.contourArea(cnt)
#     # print(area)
#     # if area >200 and area <1500:  # 过滤面积较小的噪声，可根据实际情况调整
#     if area > 60 and area < 600:  # 过滤面积较小的噪声，可根据实际情况调整
#         count += 1
#         center, radius = cv2.minEnclosingCircle(cnt)  # 求最小包围圆
#         cv2.circle(imgContours, (int(center[0]), int(center[1])), int(radius),
#                    (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)), -1)
#
# # 在图像上标记检测到的块数
# cv2.putText(imgContours, f'num: {count}', (10, 60),
#             cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
#
# plt.figure(3, figsize=(5,10))
# plt_show('Result', imgContours)
#
# plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

def plt_show(name, img, sub=111):
    """
    辅助显示函数：将BGR图像转换为RGB显示，并支持子图布局
    """
    plt.subplot(sub)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(name)
    plt.axis("off")
    # 若该行子图已满，则show()
    if (sub - 100) // 10 == (sub - 100) % 10:
        plt.show()

# 1. 读取图像
img = cv2.imread('Images/Count/seeds1.png')
if img is None:
    print("找不到图像文件，请检查路径！")
    exit()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(1, figsize=(12, 6))
plt_show('Orig img', img, 231)

# 2. 阈值分割（这里使用 Triangle 方法 + BINARY）
#   - 对于圆形、亮度相对均匀的目标，该方法通常有不错的效果
_, thresh1 = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# _, thresh1 = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
plt_show("Threshold", thresh1, 232)

# 3. 腐蚀操作，让白色区域更“确定”是前景
#    - 腐蚀可以去除一些细小毛刺或噪点，也会稍微减小目标的尺寸
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
erode = cv2.morphologyEx(thresh1, cv2.MORPH_ERODE, kernel, iterations=2)
plt_show("erode", erode, 233)

# 4. 距离变换
#    - 对二值图像做距离变换，距离越大说明越接近“中心”
dist = cv2.distanceTransform(erode, cv2.DIST_L2, 3)
#    - 归一化到 [0,1] 便于可视化
cv2.normalize(dist, dist, 0, 1, cv2.NORM_MINMAX)
plt_show("Distance", dist, 234)

# 5. 再次阈值分割距离变换图，提取“中心区域”
#    - 这里用 0.6 作为分割阈值，可根据图像实际情况调节
_, thresh2 = cv2.threshold(dist, 0.05, 1.0, cv2.THRESH_BINARY)
#    - 注意 distanceTransform 的结果是浮点型，需要转为 uint8
thresh2 = (thresh2 * 255).astype(np.uint8)
plt_show("Threshold2", thresh2, 235)

# 6. 轮廓检测，统计目标数量并用随机颜色显示
contours, hier = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
count = 0
imgContours = np.zeros_like(img)  # 创建一张全黑图像来绘制结果
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if area<1200:
        count += 1
        color = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)
        )
        cv2.drawContours(imgContours, contours, i, color, -1)  # 以填充方式绘制

print('圆形数量：', count)
cv2.putText(imgContours, f"{count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Detected Circles", imgContours)
cv2.imwrite("Detected_Circles_seeds1.png", imgContours)
# plt_show('Result', imgContours, 236)

plt.show()

