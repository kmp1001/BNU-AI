# import cv2
# import numpy as np
# import heapq  # Python内置最小堆，用于实现Dijkstra优先队列
# import matplotlib.pyplot as plt
#
#
# def dijkstra_path(binary, start, end, dt):
#     """
#     使用Dijkstra算法，在带权网格(binary)上搜索从start到end的最优路径。
#     权重由 distance transform dt 给出：cost = 1 / (dt[y,x] + 1).
#     """
#     h, w = binary.shape
#     INF = 1e9
#     dist = np.full((h, w), INF, dtype=np.float32)  # 记录到达每个像素的最小代价
#     dist[start[0], start[1]] = 0.0
#
#     parent = dict()
#     pq = []
#     heapq.heappush(pq, (0.0, start))
#
#     neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#     while pq:
#         cur_cost, (cy, cx) = heapq.heappop(pq)
#         if (cy, cx) == end:
#             break
#         if cur_cost > dist[cy, cx]:
#             continue
#         for dy, dx in neighbors:
#             ny, nx = cy + dy, cx + dx
#             if 0 <= ny < h and 0 <= nx < w:
#                 if binary[ny, nx] == 255:
#                     step_cost = 1.0 / (dt[ny, nx] + 1.0)
#                     new_cost = cur_cost + step_cost
#                     if new_cost < dist[ny, nx]:
#                         dist[ny, nx] = new_cost
#                         parent[(ny, nx)] = (cy, cx)
#                         heapq.heappush(pq, (new_cost, (ny, nx)))
#
#     if dist[end[0], end[1]] == INF:
#         return None
#
#     path = []
#     cur = end
#     while cur != start:
#         path.append(cur)
#         cur = parent[cur]
#     path.append(start)
#     path.reverse()
#     return path
#
#
# def smooth_path(path, window_size=5):
#     """
#     简单的滑动平均平滑，将原始路径平滑化。
#     """
#     if window_size < 3:
#         return path
#     half_win = window_size // 2
#     smooth_pts = []
#     for i in range(len(path)):
#         window = path[max(0, i - half_win): min(len(path), i + half_win + 1)]
#         avg_r = int(round(sum(pt[0] for pt in window) / len(window)))
#         avg_c = int(round(sum(pt[1] for pt in window) / len(window)))
#         smooth_pts.append((avg_r, avg_c))
#     return smooth_pts
#
#
# def get_closed_outer_mask(binary, morph_ksize=15, morph_iter=2):
#     """
#     对二值图做形态学操作以粘连外部粗线条，然后选取最大轮廓并填充为mask返回。
#     """
#     kernel = np.ones((morph_ksize, morph_ksize), np.uint8)
#     closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)
#
#     contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return np.zeros_like(binary, dtype=np.uint8)
#
#     max_contour = max(contours, key=cv2.contourArea)
#
#     mask = np.zeros_like(binary, dtype=np.uint8)
#     cv2.drawContours(mask, [max_contour], -1, 255, thickness=-1)
#
#     return mask
#
#
# def main(start_point_predefined=None, end_point_predefined=None):
#     # ----------------------------
#     # 1. 读取图像 & 二值化
#     # ----------------------------
#     image_path = 'Puzzle9.png'
#     original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     if original_img is None:
#         raise FileNotFoundError("无法读取图像，请检查路径。")
#
#     img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
#     gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     # 如果迷宫为“墙=白, 通路=黑”，可在这里取反
#     # binary = cv2.bitwise_not(binary)
#
#     # ----------------------------
#     # 1.1 先获取“外部轮廓”并把它闭合，得到一个mask
#     # ----------------------------
#     outer_mask = get_closed_outer_mask(binary, morph_ksize=15, morph_iter=2)
#     # 与原binary做与操作，只保留最大外部轮廓内部区域
#     # 这样就防止从外部绕路
#     final_binary = cv2.bitwise_and(binary, outer_mask)
#
#     # ----------------------------
#     # 2. 获取起点和终点
#     # ----------------------------
#     if start_point_predefined is not None and end_point_predefined is not None:
#         start_point = start_point_predefined  # (row, col)
#         end_point = end_point_predefined
#         print(f"使用预定义起点：({start_point[1]}, {start_point[0]}) 和终点：({end_point[1]}, {end_point[0]})")
#     else:
#         # 通过Matplotlib交互式点击选择
#         plt.figure(figsize=(8, 6))
#         plt.imshow(img_rgb)
#         plt.title("请点击图像中的【起点】和【终点】")
#         # ginput(2)等待用户点击2次，返回 [(x1,y1), (x2,y2)]
#         pts = plt.ginput(2, timeout=0)
#         plt.close()
#         if len(pts) < 2:
#             raise Exception("未选择足够的点，请重新运行程序。")
#         start_point = (int(round(pts[0][1])), int(round(pts[0][0])))
#         end_point = (int(round(pts[1][1])), int(round(pts[1][0])))
#         print(f"起点选择为：({start_point[1]}, {start_point[0]})")
#         print(f"终点选择为：({end_point[1]}, {end_point[0]})")
#
#     # ----------------------------
#     # 3. 计算距离变换并执行Dijkstra寻路
#     # ----------------------------
#     dt = cv2.distanceTransform(final_binary, cv2.DIST_L2, 3)
#
#     # 检查起点/终点是否在通路上
#     if final_binary[start_point[0], start_point[1]] != 255:
#         print("警告：起点不在通路上，可能导致路径搜索失败。")
#     if final_binary[end_point[0], end_point[1]] != 255:
#         print("警告：终点不在通路上，可能导致路径搜索失败。")
#
#     path = dijkstra_path(final_binary, start_point, end_point, dt)
#     if path is None:
#         raise Exception("找不到从起点到终点的路径，请确认迷宫连通性或起终点。")
#
#     # ----------------------------
#     # 4. 平滑路径并绘制结果
#     # ----------------------------
#     smoothed_path = smooth_path(path, window_size=7)
#     solution_img = original_img.copy()
#     pts = np.array([[p[1], p[0]] for p in smoothed_path], np.int32).reshape((-1, 1, 2))
#     cv2.polylines(solution_img, [pts], isClosed=False, color=(0, 255, 0), thickness=5)
#
#     solution_img_rgb = cv2.cvtColor(solution_img, cv2.COLOR_BGR2RGB)
#     plt.figure(figsize=(8, 6))
#     plt.imshow(solution_img_rgb)
#     plt.title("Dijkstra-基于距离变换的迷宫路径 (已限制外部)")
#     plt.axis('off')
#     plt.show()
#
#     cv2.imwrite("solved_outer_closed.png", solution_img)
#     print("结果已保存为 solved_outer_closed.png")
#
#
# if __name__ == "__main__":
#     # start_point_predefined = (770,220)
#     # end_point_predefined = (597,717)
#     main(start_point_predefined, end_point_predefined)
#
# import cv2
# import numpy as np
# from skimage.morphology import skeletonize
# from scipy.ndimage import distance_transform_edt
#
# # 读取图像
# image = cv2.imread('Puzzle9.png', cv2.IMREAD_GRAYSCALE)
#
# # 二值化图像
# _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
#
# # 骨架化
# skeleton = skeletonize(binary // 255)
# skeleton = (skeleton * 255).astype(np.uint8)
#
# # 计算距离变换（局部宽度）
# distance = distance_transform_edt(binary)
#
# # 创建一个空白图像用于绘制粗线条部分
# thick_lines = np.zeros_like(image)
#
# # 调整阈值，筛选粗线条部分
# thickness_threshold = 5  # 你可以尝试调小或者调大该阈值
# thick_mask = distance > thickness_threshold
# thick_lines[thick_mask] = 255
#
# # 增加一次膨胀操作，使得断裂部分更加连贯
# kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# dilated = cv2.dilate(thick_lines, kernel_dilate, iterations=1)
#
# # 使用形态学闭合操作连接粗线条部分
# kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
# connected = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
# #
# # connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel_close)
# #
# # import cv2
# # import numpy as np
# #
# #
# # def find_min_distance(cnt1, cnt2):
# #     min_dist = float('inf')
# #     closest_points = (None, None)
# #     for p1 in cnt1:
# #         for p2 in cnt2:
# #             dist = np.linalg.norm(p1[0] - p2[0])
# #             if dist < min_dist:
# #                 min_dist = dist
# #                 closest_points = (p1[0], p2[0])
# #     return min_dist, closest_points
# #
# #
# # def merge_all_contours(img, min_area=100, max_iterations=10):
# #     """
# #     不断查找轮廓并合并最相近的两条轮廓，直到轮廓数减少到1或达到最大迭代次数。
# #     img: 二值图
# #     min_area: 忽略面积小于这个值的轮廓
# #     max_iterations: 为了避免死循环，设一个最大迭代次数
# #     """
# #     for _ in range(max_iterations):
# #         # 找轮廓
# #         contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #
# #         # 过滤小噪声
# #         contours = [c for c in contours if cv2.contourArea(c) > min_area]
# #
# #         # 如果轮廓数少于2，就不用再合并了
# #         if len(contours) < 2:
# #             break
# #
# #         # 找到所有轮廓中最相近的一对
# #         min_dist = float('inf')
# #         pair = None
# #         pair_indices = None
# #
# #         for i in range(len(contours)):
# #             for j in range(i + 1, len(contours)):
# #                 d, (p1, p2) = find_min_distance(contours[i], contours[j])
# #                 if d < min_dist:
# #                     min_dist = d
# #                     pair = (p1, p2)
# #                     pair_indices = (i, j)
# #
# #         # 如果找到最相近的一对，则连线
# #         if pair is not None:
# #             cv2.line(img, tuple(pair[0]), tuple(pair[1]), 255, 2)
# #         else:
# #             # 如果没找到，就跳出
# #             break
# #
# #     return img
# # # 不断合并所有主要轮廓
# # result = merge_all_contours(connected, min_area=10, max_iterations=40)
#
# cv2.imshow('Bridged', result)
# # 显示结果
# # cv2.imshow('Thick Lines', thick_lines)
# # cv2.imshow('Dilated', dilated)
# # cv2.imshow('Connected Lines', connected)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# import cv2
# import numpy as np
# from skimage.morphology import skeletonize
# from scipy.ndimage import distance_transform_edt
#
# def find_min_distance(cnt1, cnt2):
#     """
#     在两个轮廓之间寻找最近点对，并返回最短距离和对应的两个点。
#     """
#     min_dist = float('inf')
#     closest_points = (None, None)
#     for p1 in cnt1:
#         for p2 in cnt2:
#             dist = np.linalg.norm(p1[0] - p2[0])
#             if dist < min_dist:
#                 min_dist = dist
#                 closest_points = (p1[0], p2[0])
#     return min_dist, closest_points
#
# def get_tangent_direction(contour, point, num_neighbors=10):
#     """
#     从轮廓中取离point最近的num_neighbors个点，用cv2.fitLine估计局部切线方向。
#     """
#     distances = np.linalg.norm(contour[:, 0] - point, axis=1)
#     indices = np.argsort(distances)[:num_neighbors]
#     pts = contour[indices]
#     pts = pts.reshape(-1, 1, 2)
#     [vx, vy, _, _] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
#     direction = np.array([vx, vy]).flatten()
#     norm = np.linalg.norm(direction)
#     if norm == 0:
#         return np.array([0, 0])
#     return direction / norm
#
# def cubic_bezier(P0, P1, P2, P3, num_points=100):
#     """
#     计算四次贝塞尔曲线上num_points个点，返回曲线坐标数组。
#     """
#     t = np.linspace(0, 1, num_points)[:, None]
#     curve = (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3
#     return curve.astype(np.int32)
#
# # --------------------- 图像预处理 ---------------------
# # 读取图像并转为灰度
# image = cv2.imread('Puzzle9.png', cv2.IMREAD_GRAYSCALE)
#
# # 二值化：假设目标为黑色背景、前景为白色，所以取反二值化
# _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
#
# # 可选：骨架化（这里仅作参考）
# skeleton = skeletonize(binary // 255)
# skeleton = (skeleton * 255).astype(np.uint8)
#
# # 距离变换用于判断线条粗细
# distance = distance_transform_edt(binary)
#
# # 筛选粗线条部分（局部宽度大于设定阈值）
# thick_lines = np.zeros_like(image)
# thickness_threshold = 5
# thick_mask = distance > thickness_threshold
# thick_lines[thick_mask] = 255
#
# # 膨胀操作，使断裂部分更连贯
# kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# dilated = cv2.dilate(thick_lines, kernel_dilate, iterations=1)
#
# # 闭合操作连接大部分断裂
# kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
# connected = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
#
# # --------------------- 轮廓查找与贝塞尔曲线拟合 ---------------------
# # 找出外部轮廓，并过滤掉小面积噪声
# contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = [c for c in contours if cv2.contourArea(c) > 100]
#
# if len(contours) < 2:
#     print("轮廓数量不足，不需要连接断口。")
# else:
#     # 取面积最大的两个轮廓（假设为需要连接的断口所在部分）
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     cnt1, cnt2 = contours[0], contours[1]
#
#     # 找出这两个轮廓之间最近的两个点
#     min_dist, (pt1, pt2) = find_min_distance(cnt1, cnt2)
#
#     # 分别计算这两个断口附近的局部切线方向
#     tangent1 = get_tangent_direction(cnt1, pt1, num_neighbors=10)
#     tangent2 = get_tangent_direction(cnt2, pt2, num_neighbors=10)
#
#     # 控制点距离因子，根据图像尺寸调整（例如20个像素）
#     k = 20
#     P0 = np.array(pt1, dtype=np.float32)
#     P3 = np.array(pt2, dtype=np.float32)
#     P1 = P0 + k * tangent1
#     P2 = P3 - k * tangent2
#
#     # 计算贝塞尔曲线上的点（这条曲线将补上断口）
#     bezier_curve = cubic_bezier(P0, P1, P2, P3, num_points=100)
#
#     # --------------------- 将曲线直接写入二值图 ---------------------
#     # 直接在二值图上绘制曲线，填补断口，使轮廓连续
#     connected_final = connected.copy()
#     cv2.polylines(connected_final, [bezier_curve], isClosed=False, color=255, thickness=2)
#
#     # 可选：再用闭合操作平滑细节
#     kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     connected_final = cv2.morphologyEx(connected_final, cv2.MORPH_CLOSE, kernel_final, iterations=1)
#
#     # 可选：填充闭合区域，确保形成一个完整的连通块
#     contours_final, _ = cv2.findContours(connected_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours_final:
#         largest_contour = max(contours_final, key=cv2.contourArea)
#         filled = np.zeros_like(connected_final)
#         cv2.drawContours(filled, [largest_contour], -1, 255, thickness=cv2.FILLED)
#     else:
#         filled = connected_final
#
# import cv2
# import numpy as np
# from skimage.morphology import skeletonize
# from scipy.ndimage import distance_transform_edt
# import heapq
# import matplotlib.pyplot as plt
#
# ##############################################
# # 代码1：区域提取与断口补齐相关函数
# ##############################################
# def find_min_distance(cnt1, cnt2):
#     """
#     在两个轮廓之间寻找最近点对，并返回最短距离和对应的两个点。
#     """
#     min_dist = float('inf')
#     closest_points = (None, None)
#     for p1 in cnt1:
#         for p2 in cnt2:
#             dist = np.linalg.norm(p1[0] - p2[0])
#             if dist < min_dist:
#                 min_dist = dist
#                 closest_points = (p1[0], p2[0])
#     return min_dist, closest_points
#
# def get_tangent_direction(contour, point, num_neighbors=10):
#     """
#     从轮廓中取离point最近的num_neighbors个点，用cv2.fitLine估计局部切线方向。
#     """
#     distances = np.linalg.norm(contour[:, 0] - point, axis=1)
#     indices = np.argsort(distances)[:num_neighbors]
#     pts = contour[indices]
#     pts = pts.reshape(-1, 1, 2)
#     [vx, vy, _, _] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
#     direction = np.array([vx, vy]).flatten()
#     norm = np.linalg.norm(direction)
#     if norm == 0:
#         return np.array([0, 0])
#     return direction / norm
#
# def cubic_bezier(P0, P1, P2, P3, num_points=100):
#     """
#     计算四次贝塞尔曲线上num_points个点，返回曲线坐标数组。
#     """
#     t = np.linspace(0, 1, num_points)[:, None]
#     curve = (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3
#     return curve.astype(np.int32)
#
# ##############################################
# # 代码2：迷宫寻路相关函数
# ##############################################
# def dijkstra_path(binary, start, end, dt):
#     """
#     使用Dijkstra算法，在二值图(binary)上搜索从start到end的最优路径。
#     权重由距离变换 dt 给出：cost = 1 / (dt[y,x] + 1).
#     """
#     h, w = binary.shape
#     INF = 1e9
#     dist = np.full((h, w), INF, dtype=np.float32)
#     dist[start[0], start[1]] = 0.0
#     parent = dict()
#     pq = []
#     heapq.heappush(pq, (0.0, start))
#     # 4邻域
#     neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#     while pq:
#         cur_cost, (cy, cx) = heapq.heappop(pq)
#         if (cy, cx) == end:
#             break
#         if cur_cost > dist[cy, cx]:
#             continue
#         for dy, dx in neighbors:
#             ny, nx = cy + dy, cx + dx
#             if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] == 255:
#                 step_cost = 1.0 / (dt[ny, nx] + 1.0)
#                 new_cost = cur_cost + step_cost
#                 if new_cost < dist[ny, nx]:
#                     dist[ny, nx] = new_cost
#                     parent[(ny, nx)] = (cy, cx)
#                     heapq.heappush(pq, (new_cost, (ny, nx)))
#     if dist[end[0], end[1]] == INF:
#         return None
#     path = []
#     cur = end
#     while cur != start:
#         path.append(cur)
#         cur = parent[cur]
#     path.append(start)
#     path.reverse()
#     return path
#
# def smooth_path(path, window_size=5):
#     """
#     简单的滑动平均平滑，将原始路径平滑化。
#     """
#     if window_size < 3:
#         return path
#     half_win = window_size // 2
#     smooth_pts = []
#     for i in range(len(path)):
#         window = path[max(0, i - half_win): min(len(path), i + half_win + 1)]
#         avg_r = int(round(sum(pt[0] for pt in window) / len(window)))
#         avg_c = int(round(sum(pt[1] for pt in window) / len(window)))
#         smooth_pts.append((avg_r, avg_c))
#     return smooth_pts
#
# ##############################################
# # 综合代码：先补齐区域断口，再在区域内寻路
# ##############################################
# def main():
#     # -----------------------------
#     # 1. 读取图像并预处理（代码1部分）
#     # -----------------------------
#     image_path = 'Puzzle9.png'
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         raise FileNotFoundError("无法读取图像，请检查路径。")
#
#     # 二值化（假设前景为黑色区域）
#     _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
#
#     # 可选：骨架化
#     skeleton = skeletonize(binary // 255)
#     skeleton = (skeleton * 255).astype(np.uint8)
#
#     # 距离变换，用于筛选较粗的线条
#     distance = distance_transform_edt(binary)
#     thick_lines = np.zeros_like(image)
#     thickness_threshold = 5
#     thick_mask = distance > thickness_threshold
#     thick_lines[thick_mask] = 255
#
#     # 膨胀与闭合
#     kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     dilated = cv2.dilate(thick_lines, kernel_dilate, iterations=1)
#     kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
#     connected = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
#
#     # 轮廓查找与贝塞尔曲线拟合（连接主要断口）
#     contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # 过滤掉小面积噪声
#     contours = [c for c in contours if cv2.contourArea(c) > 100]
#     if len(contours) < 2:
#         print("轮廓数量不足，不需要补齐断口。")
#         region_mask = connected.copy()
#     else:
#         # 取面积最大的两个轮廓（假设为目标区域断口处）
#         contours = sorted(contours, key=cv2.contourArea, reverse=True)
#         cnt1, cnt2 = contours[0], contours[1]
#         # 找出两轮廓间最近的点对
#         min_dist, (pt1, pt2) = find_min_distance(cnt1, cnt2)
#         # 分别求局部切线方向
#         tangent1 = get_tangent_direction(cnt1, pt1, num_neighbors=10)
#         tangent2 = get_tangent_direction(cnt2, pt2, num_neighbors=10)
#         k = 20  # 控制点延伸距离，根据实际情况调整
#         P0 = np.array(pt1, dtype=np.float32)
#         P3 = np.array(pt2, dtype=np.float32)
#         P1 = P0 + k * tangent1
#         P2 = P3 - k * tangent2
#         bezier_curve = cubic_bezier(P0, P1, P2, P3, num_points=100)
#         # 将贝塞尔曲线写入图像，补上断口
#         connected_final = connected.copy()
#         cv2.polylines(connected_final, [bezier_curve], isClosed=False, color=255, thickness=2)
#         # 用小核闭合平滑细节
#         kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         connected_final = cv2.morphologyEx(connected_final, cv2.MORPH_CLOSE, kernel_final, iterations=1)
#         # 填充闭合区域，确保形成一个完整连通块
#         contours_final, _ = cv2.findContours(connected_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours_final:
#             largest_contour = max(contours_final, key=cv2.contourArea)
#             filled = np.zeros_like(connected_final)
#             cv2.drawContours(filled, [largest_contour], -1, 255, thickness=cv2.FILLED)
#         else:
#             filled = connected_final
#         region_mask = filled  # 此为最终闭合区域（迷宫区域）
#
#     # 保存或显示区域提取结果
#     cv2.imshow("Region Mask", region_mask)
#     cv2.waitKey(1)
#
#     # -----------------------------
#     # 2. 在region_mask内做迷宫寻路（代码2部分）
#     # -----------------------------
#     # 注意：region_mask为前景=255的区域，我们认为这部分为通路，背景为墙
#     # 计算距离变换图 dt，用于衡量离墙距离
#     dt = cv2.distanceTransform(region_mask, cv2.DIST_L2, 3)
#
#     # 设置起点和终点（确保落在 region_mask 内）
#     # 此处预定义的点坐标需在提取的区域内，否则请使用交互式选点
#     start_point = (720, 210)  # 格式：(row, col)
#     end_point = (597, 700)
#
#     # 检查起点终点是否在通路内
#     if region_mask[start_point[0], start_point[1]] != 255:
#         print("警告：起点不在提取区域内。")
#     if region_mask[end_point[0], end_point[1]] != 255:
#         print("警告：终点不在提取区域内。")
#
#     # 用Dijkstra搜索区域内最优路径
#     path = dijkstra_path(region_mask, start_point, end_point, dt)
#     # 用Dijkstra搜索区域内最优路径
#     path = dijkstra_path(region_mask, start_point, end_point, dt)
#     if path is None:
#         # 可视化起点和终点在region_mask中的位置
#         vis_img = cv2.cvtColor(region_mask, cv2.COLOR_GRAY2BGR)
#         # 在起点画一个红色圆圈
#         cv2.circle(vis_img, (start_point[1], start_point[0]), 5, (0, 0, 255), -1)
#         # 在终点画一个蓝色圆圈
#         cv2.circle(vis_img, (end_point[1], end_point[0]), 5, (255, 0, 0), -1)
#         cv2.putText(vis_img, "Start", (start_point[1] + 10, start_point[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
#                     2)
#         cv2.putText(vis_img, "End", (end_point[1] + 10, end_point[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
#         cv2.imshow("Region with Start/End", vis_img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         raise Exception("在提取区域内找不到从起点到终点的路径。")
#
#     # 对路径进行平滑处理
#     smoothed_path = smooth_path(path, window_size=7)
#
#     # 将路径绘制在原图上（这里转换为BGR用于彩色显示）
#     solution_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     pts = np.array([[p[1], p[0]] for p in smoothed_path], np.int32).reshape((-1, 1, 2))
#     cv2.polylines(solution_img, [pts], isClosed=False, color=(0, 255, 0), thickness=3)
#
#     # 显示最终结果
#     cv2.imshow("Maze Path in Region", solution_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imwrite("solved_combined.png", solution_img)
#     print("最终结果已保存为 solved_combined.png")
#
# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import heapq
import matplotlib.pyplot as plt

##############################################
# 代码1：区域提取与断口补齐相关函数
##############################################
def find_min_distance(cnt1, cnt2):
    """
    在两个轮廓之间寻找最近点对，并返回最短距离和对应的两个点。
    """
    min_dist = float('inf')
    closest_points = (None, None)
    for p1 in cnt1:
        for p2 in cnt2:
            dist = np.linalg.norm(p1[0] - p2[0])
            if dist < min_dist:
                min_dist = dist
                closest_points = (p1[0], p2[0])
    return min_dist, closest_points

def get_tangent_direction(contour, point, num_neighbors=10):
    """
    从轮廓中取离 point 最近的 num_neighbors 个点，用 cv2.fitLine 估计局部切线方向。
    """
    distances = np.linalg.norm(contour[:, 0] - point, axis=1)
    indices = np.argsort(distances)[:num_neighbors]
    pts = contour[indices]
    pts = pts.reshape(-1, 1, 2)
    [vx, vy, _, _] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    direction = np.array([vx, vy]).flatten()
    norm = np.linalg.norm(direction)
    if norm == 0:
        return np.array([0, 0])
    return direction / norm

def cubic_bezier(P0, P1, P2, P3, num_points=100):
    """
    计算四次贝塞尔曲线上 num_points 个点，返回曲线坐标数组。
    """
    t = np.linspace(0, 1, num_points)[:, None]
    curve = (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3
    return curve.astype(np.int32)

##############################################
# 代码2：迷宫寻路相关函数
##############################################
def dijkstra_path(binary, start, end, dt):
    """
    使用 Dijkstra 算法，在二值图(binary)上搜索从 start 到 end 的最优路径。
    权重由距离变换 dt 给出：cost = 1 / (dt[y,x] + 1).
    """
    h, w = binary.shape
    INF = 1e9
    dist = np.full((h, w), INF, dtype=np.float32)
    dist[start[0], start[1]] = 0.0
    parent = dict()
    pq = []
    heapq.heappush(pq, (0.0, start))
    # 4邻域
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while pq:
        cur_cost, (cy, cx) = heapq.heappop(pq)
        if (cy, cx) == end:
            break
        if cur_cost > dist[cy, cx]:
            continue
        for dy, dx in neighbors:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] == 255:
                step_cost = 1.0 / (dt[ny, nx] + 1.0)
                new_cost = cur_cost + step_cost
                if new_cost < dist[ny, nx]:
                    dist[ny, nx] = new_cost
                    parent[(ny, nx)] = (cy, cx)
                    heapq.heappush(pq, (new_cost, (ny, nx)))
    if dist[end[0], end[1]] == INF:
        return None
    path = []
    cur = end
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    path.reverse()
    return path

def smooth_path(path, window_size=5):
    """
    简单的滑动平均平滑，将原始路径平滑化。
    """
    if window_size < 3:
        return path
    half_win = window_size // 2
    smooth_pts = []
    for i in range(len(path)):
        window = path[max(0, i - half_win): min(len(path), i + half_win + 1)]
        avg_r = int(round(sum(pt[0] for pt in window) / len(window)))
        avg_c = int(round(sum(pt[1] for pt in window) / len(window)))
        smooth_pts.append((avg_r, avg_c))
    return smooth_pts

##############################################
# 交互式鼠标选点相关函数
##############################################
selected_points = []
img_for_select = None

def mouse_callback(event, x, y, flags, param):
    global selected_points, img_for_select
    if event == cv2.EVENT_LBUTTONDOWN:
        pt = (y, x)  # (row, col)
        selected_points.append(pt)
        print("选中的点：", pt)
        cv2.circle(img_for_select, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img_for_select, f"{pt}", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.imshow("Select Start and End", img_for_select)

##############################################
# 综合代码：先补齐区域断口，再在区域内寻路（路径严格在内部，不接触边界）
##############################################
def main():
    global img_for_select, selected_points

    # -----------------------------
    # 1. 读取图像并预处理（区域提取与补齐）
    # -----------------------------
    image_path = 'Puzzle10.png'
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError("无法读取图像，请检查路径。")

    # 二值化：假设前景（目标线）为黑色，则取反得到白色通道
    _, binary = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY_INV)

    # 可选：骨架化
    skeleton = skeletonize(binary // 255)
    skeleton = (skeleton * 255).astype(np.uint8)

    # 距离变换，用于筛选较粗的线条
    distance = distance_transform_edt(binary)
    thick_lines = np.zeros_like(image_gray)
    thickness_threshold = 5
    thick_mask = distance > thickness_threshold
    thick_lines[thick_mask] = 255

    # 膨胀与闭合，得到初步的闭合边界
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(thick_lines, kernel_dilate, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    connected = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)

    # 轮廓查找与贝塞尔曲线拟合（连接主要断口）
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 100]
    if len(contours) < 2:
        print("轮廓数量不足，不需要补齐断口。")
        region_mask = connected.copy()
    else:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cnt1, cnt2 = contours[0], contours[1]
        min_dist, (pt1, pt2) = find_min_distance(cnt1, cnt2)
        tangent1 = get_tangent_direction(cnt1, pt1, num_neighbors=10)
        tangent2 = get_tangent_direction(cnt2, pt2, num_neighbors=10)
        k = 20
        P0 = np.array(pt1, dtype=np.float32)
        P3 = np.array(pt2, dtype=np.float32)
        P1 = P0 + k * tangent1
        P2 = P3 - k * tangent2
        bezier_curve = cubic_bezier(P0, P1, P2, P3, num_points=100)
        connected_final = connected.copy()
        cv2.polylines(connected_final, [bezier_curve], isClosed=False, color=255, thickness=2)
        kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        connected_final = cv2.morphologyEx(connected_final, cv2.MORPH_CLOSE, kernel_final, iterations=1)
        contours_final, _ = cv2.findContours(connected_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_final:
            largest_contour = max(contours_final, key=cv2.contourArea)
            filled = np.zeros_like(connected_final)
            cv2.drawContours(filled, [largest_contour], -1, 255, thickness=cv2.FILLED)
        else:
            filled = connected_final
        region_mask = filled

    # 显示区域提取结果（闭合边界）
    cv2.imshow("Region Mask", region_mask)
    cv2.waitKey(1)

    # -----------------------------
    # 2. 对 filled 进行腐蚀，得到 safe_mask（确保路径不触及边界）
    # -----------------------------
    # 这里用一个较小的椭圆核进行腐蚀，收缩边界一定像素
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    safe_mask = cv2.erode(region_mask, kernel_erode, iterations=1)

    # -----------------------------
    # 3. 交互式选点（在原始彩色图与闭合区域半透明叠加图上选点）
    # -----------------------------
    orig_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if orig_color is None:
        raise FileNotFoundError("无法读取原始彩色图像。")
    # 将 safe_mask 区域用半透明深黄色叠加
    mask_color = np.zeros_like(orig_color)
    mask_color[safe_mask == 255] = (0, 215, 255)  # 深黄色
    alpha = 0.3
    img_for_select = cv2.addWeighted(orig_color, 1, mask_color, alpha, 0)

    # 创建选点窗口并绑定鼠标回调
    cv2.namedWindow("Select Start and End")
    cv2.setMouseCallback("Select Start and End", mouse_callback)
    print("请在弹出窗口中依次选择起点和终点（要求在内部安全区域内）...")
    while True:
        cv2.imshow("Select Start and End", img_for_select)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(selected_points) >= 2:
            break
    cv2.destroyWindow("Select Start and End")

    if len(selected_points) < 2:
        raise Exception("未选择足够的点，请重新运行程序。")
    start_point, end_point = selected_points[:2]
    print("起点：", start_point)
    print("终点：", end_point)

    # 检查起点终点是否在 safe_mask 内
    if safe_mask[start_point[0], start_point[1]] != 255:
        print("警告：起点不在安全区域内。")
    if safe_mask[end_point[0], end_point[1]] != 255:
        print("警告：终点不在安全区域内。")

    # -----------------------------
    # 4. 在 safe_mask 内做迷宫寻路
    # -----------------------------
    # 计算 safe_mask 内的距离变换图 dt
    dt = cv2.distanceTransform(safe_mask, cv2.DIST_L2, 3)
    path = dijkstra_path(safe_mask, start_point, end_point, dt)
    if path is None:
        vis_img = cv2.cvtColor(safe_mask, cv2.COLOR_GRAY2BGR)
        cv2.circle(vis_img, (start_point[1], start_point[0]), 5, (0, 0, 255), -1)
        cv2.circle(vis_img, (end_point[1], end_point[0]), 5, (255, 0, 0), -1)
        cv2.putText(vis_img, "Start", (start_point[1]+10, start_point[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.putText(vis_img, "End", (end_point[1]+10, end_point[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        cv2.imshow("Safe Region with Start/End", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        raise Exception("在安全区域内找不到从起点到终点的路径。")

    smoothed_path = smooth_path(path, window_size=7)

    # -----------------------------
    # 5. 显示结果：构建背景（内部区域为深黄色）并绘制路径（绿色）
    # -----------------------------
    maze_bg = np.zeros((safe_mask.shape[0], safe_mask.shape[1], 3), dtype=np.uint8)
    maze_bg[safe_mask == 255] = (0, 215, 255)  # 深黄色区域
    pts = np.array([[p[1], p[0]] for p in smoothed_path], np.int32).reshape((-1, 1, 2))
    cv2.polylines(maze_bg, [pts], isClosed=False, color=(0, 255, 0), thickness=3)

    cv2.imshow("Maze Path Inside Safe Region", maze_bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("solved_inside.png", maze_bg)
    print("最终结果已保存为 solved_inside.png")

if __name__ == "__main__":
    main()
