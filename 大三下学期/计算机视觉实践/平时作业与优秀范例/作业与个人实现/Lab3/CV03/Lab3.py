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
#
#     :param binary: 迷宫二值图，通路=255，墙=0
#     :param start: 起点 (row, col)
#     :param end:   终点 (row, col)
#     :param dt:    距离变换图，与binary大小一致
#     :return:      path (list of (row, col)) 或 None 表示无法到达
#     """
#     h, w = binary.shape
#     INF = 1e9
#     dist = np.full((h, w), INF, dtype=np.float32)  # 记录到达每个像素的最小代价
#     dist[start[0], start[1]] = 0.0
#
#     parent = dict()  # 用于回溯路径
#     # 使用最小堆存放 (代价, (row, col))
#     pq = []
#     heapq.heappush(pq, (0.0, start))
#
#     # 4邻域（可改为8邻域）
#     neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#
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
#                     # cost = 1/(dt[ny, nx] + 1)
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
#     :param path: list of (row, col)
#     :param window_size: 平滑窗口大小（建议奇数）
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
# def main(start_point_predefined=None, end_point_predefined=None):
#     # ----------------------------
#     # 1. 读取图像 & 二值化
#     # ----------------------------
#     image_path = '2.png'
#     original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     if original_img is None:
#         raise FileNotFoundError("无法读取图像，请检查路径。")
#
#     # 转换为RGB用于Matplotlib显示
#     img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
#
#     gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     # 如果迷宫为“墙=白, 通路=黑”，请先取反：
#     # binary = cv2.bitwise_not(binary)
#
#     # 计算距离变换图 dt，用于衡量离墙距离
#     dt = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
#
#     # ----------------------------
#     # 2. 获取起点和终点
#     # ----------------------------
#     if start_point_predefined is not None and end_point_predefined is not None:
#         start_point = start_point_predefined  # 格式：(row, col)
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
#         # 转换为 (row, col)
#         start_point = (int(round(pts[0][1])), int(round(pts[0][0])))
#         end_point = (int(round(pts[1][1])), int(round(pts[1][0])))
#         print(f"起点选择为：({start_point[1]}, {start_point[0]})")
#         print(f"终点选择为：({end_point[1]}, {end_point[0]})")
#
#     # 检查所选点是否在通路上（白色区域）
#     if binary[start_point[0], start_point[1]] != 255:
#         print("警告：起点不在通路上，可能导致路径搜索失败。")
#     if binary[end_point[0], end_point[1]] != 255:
#         print("警告：终点不在通路上，可能导致路径搜索失败。")
#
#     # ----------------------------
#     # 3. 用Dijkstra搜索“离墙距离最大化”的路径
#     # ----------------------------
#     path = dijkstra_path(binary, start_point, end_point, dt)
#     if path is None:
#         raise Exception("找不到从起点到终点的路径，请确认迷宫连通性。")
#
#     # ----------------------------
#     # 4. 平滑路径并绘制结果
#     # ----------------------------
#     smoothed_path = smooth_path(path, window_size=7)
#
#     # 在原图上绘制平滑且较粗的绿色路径
#     solution_img = original_img.copy()
#     pts = np.array([[p[1], p[0]] for p in smoothed_path], np.int32).reshape((-1, 1, 2))
#     cv2.polylines(solution_img, [pts], isClosed=False, color=(0, 255, 0), thickness=5)
#
#     # 转换为RGB后使用Matplotlib展示
#     solution_img_rgb = cv2.cvtColor(solution_img, cv2.COLOR_BGR2RGB)
#     plt.figure(figsize=(8, 6))
#     plt.imshow(solution_img_rgb)
#     plt.title("Dijkstra")
#     plt.axis('off')
#     plt.show()
#
#     cv2.imwrite("solved_001.png", solution_img)
#     print("结果已保存为 solved_08.png")
#
# def find_red_points(image):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
#     lower_red1 = np.array([0, 100, 100])
#     upper_red1 = np.array([10, 255, 255])
#     lower_red2 = np.array([160, 100, 100])
#     upper_red2 = np.array([180, 255, 255])
#
#     mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
#     mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
#     mask = cv2.bitwise_or(mask1, mask2)
#
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     if len(contours) < 2:
#         raise Exception("未找到足够的红色区域作为起点和终点！")
#
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
#
#     points = []
#     for cnt in contours:
#         M = cv2.moments(cnt)
#         if M["m00"] == 0:
#             raise Exception("检测到的红色区域质心无效！")
#         cx = int(M["m10"] / M["m00"])
#         cy = int(M["m01"] / M["m00"])
#         points.append((cy, cx))
#
#     return points[0], points[1]
#
# if __name__ == "__main__":
#     image_path = '2.png'
#     original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     if original_img is None:
#         raise FileNotFoundError("无法读取图像，请检查路径。");
#
#     start_point_predefined, end_point_predefined = find_red_points(original_img)
#     print(f"自动检测到起点：({start_point_predefined[1]}, {start_point_predefined[0]})")
#     print(f"自动检测到终点：({end_point_predefined[1]}, {end_point_predefined[0]})")
#
#     main(start_point_predefined, end_point_predefined)
#
#
# import numpy as np
# from collections import deque
# import cv2
#
# def bfs_path(binary, start, end):
#     """在二值图上用BFS寻找路径"""
#     h, w = binary.shape
#     dist = np.full((h, w), -1, dtype=np.int32)
#     dist[start[0], start[1]] = 0
#
#     parent = {}
#     queue = deque([start])
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#
#     while queue:
#         cy, cx = queue.popleft()
#         if (cy, cx) == end:
#             break
#         for dy, dx in directions:
#             ny, nx = cy + dy, cx + dx
#             if 0 <= ny < h and 0 <= nx < w:
#                 if binary[ny, nx] == 255 and dist[ny, nx] == -1:
#                     dist[ny, nx] = dist[cy, cx] + 1
#                     parent[(ny, nx)] = (cy, cx)
#                     queue.append((ny, nx))
#
#     if dist[end[0], end[1]] == -1:
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
# def on_mouse(event, x, y, flags, param):
#     """鼠标回调函数"""
#     global points, display_img, maze_paths
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # 只允许在迷宫通路上选点
#         if maze_paths[y, x] == 255:
#             points.append((y, x))
#             cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
#             cv2.imshow("选择起点和终点", display_img)
#             if len(points) == 1:
#                 print(f"起点: ({y},{x})")
#             elif len(points) == 2:
#                 print(f"终点: ({y},{x})")
#         else:
#             print("选择的位置不是通路，请重选")
#
#
# def main():
#     global points, display_img, maze_paths
#
#     # 读取图像
#     image_path = "Puzzle1.png"
#     original = cv2.imread(image_path)
#     if original is None:
#         raise FileNotFoundError("找不到图片")
#
#     # 转灰度并二值化
#     gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
#
#     # =============================================
#     # 关键修改：手动创建气球内部掩码
#     # =============================================
#
#     # 1. 创建一个空白掩码
#     h, w = binary.shape
#     balloon_mask = np.zeros((h, w), dtype=np.uint8)
#
#     # 2. 手动绘制气球形状（从图中提取轮廓）
#     # 下面数值需要根据实际图像调整
#     balloon_center = (w // 2, h // 2)  # 气球中心点
#     balloon_radius = min(w, h) // 2 - 10  # 气球半径
#
#     # 绘制气球圆形区域
#     cv2.circle(balloon_mask, balloon_center, balloon_radius, 255, -1)
#
#     # 3. 将气球与迷宫线条叠加
#     # 从原始二值图获取迷宫线条（墙壁为黑色）
#     maze_walls = cv2.bitwise_not(binary)  # 反转，使墙壁变为白色
#
#     # 加粗墙壁以防止路径穿墙
#     kernel = np.ones((3, 3), np.uint8)
#     thick_walls = cv2.dilate(maze_walls, kernel, iterations=1)
#
#     # 从气球掩码中去除墙壁
#     maze_paths = cv2.bitwise_and(balloon_mask, cv2.bitwise_not(thick_walls))
#
#     # 保存处理后的图像以便调试
#     cv2.imwrite("debug_balloon_mask.png", balloon_mask)
#     cv2.imwrite("debug_maze_walls.png", thick_walls)
#     cv2.imwrite("debug_maze_paths.png", maze_paths)
#
#     # 交互式选择起点和终点
#     points = []
#     display_img = original.copy()
#
#     cv2.namedWindow("选择起点和终点", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("选择起点和终点", 800, 800)
#     cv2.setMouseCallback("选择起点和终点", on_mouse)
#
#     # 显示处理后的掩码，便于选点
#     cv2.namedWindow("迷宫通路", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("迷宫通路", 800, 800)
#     cv2.imshow("迷宫通路", maze_paths)
#
#     print("请在迷宫通路（白色区域）上依次点击起点和终点，然后按ESC")
#
#     while True:
#         cv2.imshow("选择起点和终点", display_img)
#         k = cv2.waitKey(1) & 0xFF
#         if k == 27 or len(points) >= 2:  # ESC或选择了两个点
#             break
#
#     if len(points) < 2:
#         print("未选择足够的点")
#         return
#
#     # 寻路
#     path = bfs_path(maze_paths, points[0], points[1])
#
#     if path is None:
#         print("找不到路径")
#     else:
#         print(f"找到路径，长度={len(path)}")
#
#         # 在原图上绘制路径
#         result = original.copy()
#         for i in range(len(path) - 1):
#             y1, x1 = path[i]
#             y2, x2 = path[i + 1]
#             cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
#
#         # 标记起点和终点
#         cv2.circle(result, (points[0][1], points[0][0]), 5, (0, 255, 0), -1)
#         cv2.circle(result, (points[1][1], points[1][0]), 5, (255, 0, 0), -1)
#
#         # 显示结果
#         cv2.namedWindow("迷宫路径", cv2.WINDOW_NORMAL)
#         cv2.resizeWindow("迷宫路径", 800, 800)
#         cv2.imshow("迷宫路径", result)
#         cv2.waitKey(0)
#
#         # 保存结果
#         cv2.imwrite("0301.png", result)
#         print("已保存结果为 maze_solution.png")
#
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()







import cv2
import numpy as np
import heapq
import matplotlib.pyplot as plt


def dijkstra_path(binary, start, end, dt):
    h, w = binary.shape
    INF = 1e9
    dist = np.full((h, w), INF, dtype=np.float32)
    dist[start[0], start[1]] = 0.0
    parent = dict()
    pq = []
    heapq.heappush(pq, (0.0, start))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while pq:
        cur_cost, (cy, cx) = heapq.heappop(pq)
        if (cy, cx) == end:
            break
        if cur_cost > dist[cy, cx]:
            continue
        for dy, dx in neighbors:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                if binary[ny, nx] == 255:
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

def find_red_points(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        raise Exception("未找到足够的红色区域作为起点和终点！")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    points = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            raise Exception("检测到的红色区域质心无效！")
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        points.append((cy, cx))  # (row, col)

    return points[0], points[1]  # 只返回两个点


def main():
    # 读取图像
    image_path = 'Puzzle10.png'
    original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_img is None:
        raise FileNotFoundError("无法读取图像，请检查路径。")

    # 自动找到红色的起点终点
    start_point, end_point = find_red_points(original_img)
    print(f"自动检测到起点：{start_point}, 终点：{end_point}")

    # 提前把红色点替换成白色，确保不会影响路径搜索

    hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    original_img[mask_red > 0] = [255, 255, 255]

    # 转灰度再二值化
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 计算距离变换图 dt
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 3)

    # 检查起点和终点是否位于通路上
    if binary[start_point[0], start_point[1]] != 255:
        print("起点不在通路上，可能导致路径搜索失败。")
    if binary[end_point[0], end_point[1]] != 255:
        print("终点不在通路上，可能导致路径搜索失败。")

    # Dijkstra算法
    path = dijkstra_path(binary, start_point, end_point, dt)
    if path is None:
        raise Exception("未找到路径")

    # 平滑路径并绘制结果
    smoothed_path = smooth_path(path, window_size=7)

    solution_img = original_img.copy()
    pts = np.array([[p[1], p[0]] for p in smoothed_path], np.int32).reshape((-1, 1, 2))
    cv2.polylines(solution_img, [pts], False, (0, 255, 0), thickness=5)

    solution_img_rgb = cv2.cvtColor(solution_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(solution_img_rgb)
    plt.title("自动识别红点作为起终点")
    plt.axis('off')
    plt.show()




if __name__ == "__main__":
    main()
