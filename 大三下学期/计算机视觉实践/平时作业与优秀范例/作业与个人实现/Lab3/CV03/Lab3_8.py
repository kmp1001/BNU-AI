import cv2
import numpy as np
import heapq  # Python内置最小堆，用于实现Dijkstra优先队列

def dijkstra_path(binary, start, end, dt):
    """
    使用Dijkstra算法，在带权网格(binary)上搜索从start到end的最优路径。
    权重由 distance transform dt 给出：cost = 1 / (dt[y,x] + 1).

    :param binary: 迷宫二值图，通路=255，墙=0
    :param start: 起点 (row, col)
    :param end:   终点 (row, col)
    :param dt:    距离变换图，与binary大小一致
    :return:      path (list of (row, col)) 或 None 表示无法到达
    """
    h, w = binary.shape
    INF = 1e9
    dist = np.full((h, w), INF, dtype=np.float32)  # 记录到达每个像素的最小代价
    dist[start[0], start[1]] = 0.0

    parent = dict()  # 用于回溯路径
    pq = []
    heapq.heappush(pq, (0.0, start))

    # 仅4邻域示例，若需要可改8邻域
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
                # 必须在通路上
                if binary[ny, nx] == 255:
                    # cost = 1/(dt[ny, nx] + 1)，离墙越远 dt越大，cost越小
                    step_cost = 1.0 / (dt[ny, nx] + 1.0)
                    new_cost = cur_cost + step_cost
                    if new_cost < dist[ny, nx]:
                        dist[ny, nx] = new_cost
                        parent[(ny, nx)] = (cy, cx)
                        heapq.heappush(pq, (new_cost, (ny, nx)))

    if dist[end[0], end[1]] == INF:
        return None

    # 回溯
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
    简单滑动平均平滑，将原始路径平滑化。
    """
    if window_size < 3:
        return path
    half_win = window_size // 2
    smoothed = []
    for i in range(len(path)):
        window = path[max(0, i - half_win): min(len(path), i + half_win + 1)]
        avg_r = int(round(sum(pt[0] for pt in window) / len(window)))
        avg_c = int(round(sum(pt[1] for pt in window) / len(window)))
        smoothed.append((avg_r, avg_c))
    return smoothed

if __name__ == "__main__":
    # ----------------------------
    # 1. 读取彩色图像
    # ----------------------------
    image_path = 'Puzzle8.png'  # 示例：黄色迷宫+黑线
    original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_img is None:
        raise FileNotFoundError("无法读取图像，请检查路径。")

    # ----------------------------
    # 2. 提取黄色区域 (通道)
    # ----------------------------
    hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    # 需根据实际图片调整的黄色范围
    lower_yellow = np.array([0, 200, 200])
    upper_yellow = np.array([50, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 可选形态学操作，去除噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel, iterations=1)

    # ----------------------------
    # 3. 提取黑色线条 (墙)
    # ----------------------------
    # 下面黑色范围也需根据实际图像调整
    # 比如 H 全部范围都可以, S和V在一定区间内
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([50, 255, 200])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # 如果黑线过细，可以适当膨胀，让墙线更粗
    mask_black = cv2.dilate(mask_black, kernel, iterations=1)

    # ----------------------------
    # 4. 从黄色通道中排除黑线
    # ----------------------------
    # 这样黑线区域就变成0，不可行走
    mask_black_inv = cv2.bitwise_not(mask_black)  # 黑线=0, 非黑线=255
    final_mask = cv2.bitwise_and(mask_yellow, mask_black_inv)
    # final_mask中 255=通道(黄)且非黑线, 0=墙(黑线)或背景

    # ----------------------------
    # 5. 如果需要保留最大轮廓（如菠萝外形），可以在此再做一次轮廓筛选
    # ----------------------------
    # (略) 与之前相同，用 findContours -> maxContour -> drawContours -> bitwise_and
    # final_mask = bitwise_and(final_mask, largest_mask)

    # ----------------------------
    # 6. 选择起点和终点
    # ----------------------------
    points = []
    display_img = original_img.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((y, x))  # (row, col)
            cv2.circle(display_img, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow("选择起点和终点", display_img)
            if len(points) == 1:
                print(f"起点已选择: ({x}, {y})")
            elif len(points) == 2:
                print(f"终点已选择: ({x}, {y})")

    cv2.namedWindow("选择起点和终点", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("选择起点和终点", 800, 600)
    cv2.setMouseCallback("选择起点和终点", mouse_callback)

    print("请在图像上依次点击【起点】和【终点】，按ESC键退出。")
    while len(points) < 2:
        cv2.imshow("选择起点和终点", display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cv2.destroyWindow("选择起点和终点")

    if len(points) < 2:
        raise Exception("未选择足够的点，请重新运行并选择起点、终点。")

    start_point = points[0]
    end_point = points[1]

    # ----------------------------
    # 7. 检查起点终点是否在通路上
    # ----------------------------
    if final_mask[start_point[0], start_point[1]] != 255:
        print("警告：起点不在通路上，可能搜索失败。")
    if final_mask[end_point[0], end_point[1]] != 255:
        print("警告：终点不在通路上，可能搜索失败。")

    # ----------------------------
    # 8. 距离变换 + Dijkstra
    # ----------------------------
    dist_transform = cv2.distanceTransform(final_mask, cv2.DIST_L2, 3)
    path = dijkstra_path(final_mask, start_point, end_point, dist_transform)
    if path is None:
        raise Exception("找不到从起点到终点的路径，可能通路不连通。")

    # ----------------------------
    # 9. 平滑路径并可视化
    # ----------------------------
    smoothed_path = smooth_path(path, window_size=7)

    solution_img = original_img.copy()
    pts = np.array([[p[1], p[0]] for p in smoothed_path], np.int32).reshape((-1, 1, 2))
    cv2.polylines(solution_img, [pts], isClosed=False, color=(0, 255, 0), thickness=5)

    cv2.namedWindow("最终迷宫路径", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("最终迷宫路径", 800, 600)
    cv2.imshow("最终迷宫路径", solution_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("solved_9", solution_img)
    print("结果已保存为 solved_color_maze.png")

#
# import cv2
# import numpy as np
# import heapq  # Python内置最小堆，用于实现Dijkstra优先队列
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
#     dist = np.full((h, w), INF, dtype=np.float32)
#     dist[start[0], start[1]] = 0.0
#
#     parent = {}
#     pq = []
#     heapq.heappush(pq, (0.0, start))
#
#     neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4邻域
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
#                 if binary[ny, nx] == 255:  # 通路
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
#     # 回溯
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
#     简单滑动平均平滑，将原始路径平滑化。
#     """
#     if window_size < 3:
#         return path
#     half_win = window_size // 2
#     smoothed = []
#     for i in range(len(path)):
#         window = path[max(0, i - half_win): min(len(path), i + half_win + 1)]
#         avg_r = int(round(sum(pt[0] for pt in window) / len(window)))
#         avg_c = int(round(sum(pt[1] for pt in window) / len(window)))
#         smoothed.append((avg_r, avg_c))
#     return smoothed
#
# if __name__ == "__main__":
#     image_path = 'Puzzle8.png'
#     original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     if original_img is None:
#         raise FileNotFoundError("无法读取图像，请检查路径。")
#
#     # ----------------------------
#     # 1. 提取初步通路 (与现有方法相同)
#     # ----------------------------
#     hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
#
#     # 示例：提取白色/浅色区域当做通路
#     lower_white = np.array([0, 0, 200])   # 需根据实际情况调整
#     upper_white = np.array([0, 0, 255])
#     mask_white = cv2.inRange(hsv, lower_white, upper_white)
#
#     # 提取黑色线条(墙)
#     lower_black = np.array([0, 0, 0])
#     upper_black = np.array([0, 0, 100])  # 可适当调整
#     mask_black = cv2.inRange(hsv, lower_black, upper_black)
#
#     # 如果黑线太细，可膨胀一下
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     mask_black = cv2.dilate(mask_black, kernel, iterations=1)
#
#     # 组合得到通路：white区域 & 非black
#     mask_black_inv = cv2.bitwise_not(mask_black)
#     final_mask = cv2.bitwise_and(mask_white, mask_black_inv)
#
#     # ----------------------------
#     # 2. 形态学闭运算，封闭外壁裂缝
#     # ----------------------------
#     kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
#     closed_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
#
#     # ----------------------------
#     # 3. 提取最大轮廓，仅保留该轮廓内部
#     # ----------------------------
#     contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         raise Exception("未找到任何轮廓，请检查颜色范围或形态学参数。")
#
#     max_contour = max(contours, key=cv2.contourArea)
#     largest_mask = np.zeros_like(closed_mask)
#     cv2.drawContours(largest_mask, [max_contour], -1, 255, thickness=cv2.FILLED)
#
#     # 仅保留最大轮廓内的通路
#     final_mask2 = cv2.bitwise_and(closed_mask, largest_mask)
#
#     # ----------------------------
#     # 4. 可选：Flood Fill 去除外部
#     # ----------------------------
#     # # 如果依旧可能从外面绕行，可在四角做 floodFill
#     # h, w = final_mask2.shape
#     # mask_flood = np.zeros((h+2, w+2), np.uint8)
#     # corners = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
#     # for (cy, cx) in corners:
#     #     if final_mask2[cy, cx] == 255:
#     #         cv2.floodFill(final_mask2, mask_flood, (cx, cy), 0)
#
#     # ----------------------------
#     # 5. 鼠标选起点终点
#     # ----------------------------
#     points = []
#     display_img = original_img.copy()
#
#     def mouse_callback(event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             points.append((y, x))  # (row, col)
#             cv2.circle(display_img, (x, y), 3, (0, 0, 255), -1)
#             cv2.imshow("选择起点和终点", display_img)
#             if len(points) == 1:
#                 print(f"起点已选择: ({x}, {y})")
#             elif len(points) == 2:
#                 print(f"终点已选择: ({x}, {y})")
#
#     cv2.namedWindow("选择起点和终点", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("选择起点和终点", 800, 600)
#     cv2.setMouseCallback("选择起点和终点", mouse_callback)
#
#     print("请在图像上依次点击【起点】和【终点】，按ESC退出。")
#     while len(points) < 2:
#         cv2.imshow("选择起点和终点", display_img)
#         key = cv2.waitKey(1) & 0xFF
#         if key == 27:
#             break
#     cv2.destroyWindow("选择起点和终点")
#
#     if len(points) < 2:
#         raise Exception("未选择足够的点，请重新运行并选择起点、终点。")
#
#     start_point = points[0]
#     end_point = points[1]
#
#     # 检查起点终点是否在通路上
#     if final_mask2[start_point[0], start_point[1]] != 255:
#         print("警告：起点不在通路上，可能搜索失败。")
#     if final_mask2[end_point[0], end_point[1]] != 255:
#         print("警告：终点不在通路上，可能搜索失败。")
#
#     # ----------------------------
#     # 6. 距离变换 + Dijkstra
#     # ----------------------------
#     dist_transform = cv2.distanceTransform(final_mask2, cv2.DIST_L2, 3)
#     path = dijkstra_path(final_mask2, start_point, end_point, dist_transform)
#     if path is None:
#         raise Exception("找不到从起点到终点的路径，可能通路不连通。")
#
#     # ----------------------------
#     # 7. 平滑并可视化
#     # ----------------------------
#     smoothed_path = smooth_path(path, window_size=7)
#     solution_img = original_img.copy()
#     pts = np.array([[p[1], p[0]] for p in smoothed_path], np.int32).reshape((-1, 1, 2))
#     cv2.polylines(solution_img, [pts], isClosed=False, color=(0, 255, 0), thickness=5)
#
#     cv2.namedWindow("最终迷宫路径", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("最终迷宫路径", 800, 600)
#     cv2.imshow("最终迷宫路径", solution_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     cv2.imwrite("solved_8.png", solution_img)
#     print("结果已保存为 solved_9.png")

#
# import cv2
# import numpy as np
# import heapq  # Python内置最小堆，用于实现Dijkstra优先队列
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
#     dist = np.full((h, w), INF, dtype=np.float32)
#     dist[start[0], start[1]] = 0.0
#
#     parent = {}
#     pq = []
#     heapq.heappush(pq, (0.0, start))
#
#     neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4邻域
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
#                 # 必须在通路上
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
#     # 回溯
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
#     简单滑动平均平滑，将原始路径平滑化。
#     """
#     if window_size < 3:
#         return path
#     half_win = window_size // 2
#     smoothed = []
#     for i in range(len(path)):
#         window = path[max(0, i - half_win): min(len(path), i + half_win + 1)]
#         avg_r = int(round(sum(pt[0] for pt in window) / len(window)))
#         avg_c = int(round(sum(pt[1] for pt in window) / len(window)))
#         smoothed.append((avg_r, avg_c))
#     return smoothed
#
# if __name__ == "__main__":
#     # ----------------------------
#     # 1. 读取图像
#     # ----------------------------
#     image_path = 'Puzzle9.png'
#     original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     if original_img is None:
#         raise FileNotFoundError("无法读取图像，请检查路径。")
#
#     hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
#
#     # ----------------------------
#     # 2. 提取白色通路
#     # ----------------------------
#     # 这个区间需要根据实际图像调试
#     # 让S最大到 40、V最小到 200，以包含白色或浅灰色
#     lower_white = np.array([0, 0, 200])
#     upper_white = np.array([0,0, 255])
#     mask_white = cv2.inRange(hsv, lower_white, upper_white)
#
#     # 提取黑色线条(墙)
#     # 同理，根据实际图像需要调节
#     lower_black = np.array([0, 0, 0])
#     upper_black = np.array([0, 0, 100])
#     mask_black = cv2.inRange(hsv, lower_black, upper_black)
#
#     # 形态学操作，让黑线更粗
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     mask_black = cv2.dilate(mask_black, kernel, iterations=1)
#
#     # 通路 = 白色 & 非黑线
#     mask_black_inv = cv2.bitwise_not(mask_black)
#     final_mask = cv2.bitwise_and(mask_white, mask_black_inv)
#
#     # ----------------------------
#     # 3. 闭运算，填补外壁裂缝
#     # ----------------------------
#     kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
#     closed_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
#
#     # ----------------------------
#     # 4. 提取最大轮廓
#     # ----------------------------
#     contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         raise Exception("未找到任何轮廓，请检查颜色范围或形态学参数。")
#
#     max_contour = max(contours, key=cv2.contourArea)
#     largest_mask = np.zeros_like(closed_mask)
#     cv2.drawContours(largest_mask, [max_contour], -1, 255, thickness=cv2.FILLED)
#
#     final_mask2 = cv2.bitwise_and(closed_mask, largest_mask)
#
#     # ----------------------------
#     # 5. 可选：Flood Fill 去除外部
#     # ----------------------------
#     # 若仍有外部连通，可对四角 floodFill
#     h, w = final_mask2.shape
#     mask_flood = np.zeros((h+2, w+2), np.uint8)
#     corners = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
#     for (cy, cx) in corners:
#         if final_mask2[cy, cx] == 255:
#             cv2.floodFill(final_mask2, mask_flood, (cx, cy), 0)
#
#     # ----------------------------
#     # 6. 鼠标选点，必须点在通路上
#     # ----------------------------
#     points = []
#     display_img = original_img.copy()
#
#     def mouse_callback(event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             # 检查点击位置是否在通路上
#             if final_mask2[y, x] == 255:
#                 points.append((y, x))
#                 cv2.circle(display_img, (x, y), 3, (0, 0, 255), -1)
#                 cv2.imshow("选择起点和终点", display_img)
#                 if len(points) == 1:
#                     print(f"起点已选择: ({x}, {y})")
#                 elif len(points) == 2:
#                     print(f"终点已选择: ({x}, {y})")
#             else:
#                 print("你点击的位置不在通路上，请重新点击！")
#
#     cv2.namedWindow("选择起点和终点", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("选择起点和终点", 800, 600)
#     cv2.setMouseCallback("选择起点和终点", mouse_callback)
#     cv2.imwrite("debug_mask_white.png", mask_white)
#     cv2.imwrite("debug_mask_black.png", mask_black)
#     cv2.imwrite("debug_final_mask.png", final_mask)
#     cv2.imwrite("debug_closed_mask.png", closed_mask)
#     cv2.imwrite("debug_largest_mask.png", largest_mask)
#     cv2.imwrite("debug_final_mask2.png", final_mask2)
#
#
#     print("请在图像上依次点击【起点】和【终点】（都要点在白色通路上），按ESC退出。")
#     while len(points) < 2:
#         cv2.imshow("选择起点和终点", display_img)
#         key = cv2.waitKey(1) & 0xFF
#         if key == 27:
#             break
#
#     cv2.destroyWindow("选择起点和终点")
#
#     if len(points) < 2:
#         raise Exception("未选择足够的点，请重新运行并选择起点、终点。")
#
#     start_point = points[0]
#     end_point = points[1]
#
#     # ----------------------------
#     # 7. 距离变换 + Dijkstra
#     # ----------------------------
#     dist_transform = cv2.distanceTransform(final_mask2, cv2.DIST_L2, 3)
#     path = dijkstra_path(final_mask2, start_point, end_point, dist_transform)
#     if path is None:
#         raise Exception("找不到从起点到终点的路径，可能通路不连通。")
#
#     # ----------------------------
#     # 8. 平滑并可视化
#     # ----------------------------
#     smoothed_path = smooth_path(path, window_size=7)
#     solution_img = original_img.copy()
#     pts = np.array([[p[1], p[0]] for p in smoothed_path], np.int32).reshape((-1, 1, 2))
#     cv2.polylines(solution_img, [pts], isClosed=False, color=(0, 255, 0), thickness=5)
#
#     cv2.namedWindow("最终迷宫路径", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("最终迷宫路径", 800, 600)
#     cv2.imshow("最终迷宫路径", solution_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#

import cv2
import numpy as np
import heapq  # Python内置最小堆，用于实现Dijkstra优先队列

def dijkstra_path(binary, start, end, dt):
    """
    使用Dijkstra算法，在带权网格(binary)上搜索从start到end的最优路径。
    权重由 distance transform dt 给出：cost = 1 / (dt[y,x] + 1).

    :param binary: 迷宫二值图，通路=255，墙=0
    :param start: 起点 (row, col)
    :param end:   终点 (row, col)
    :param dt:    距离变换图，与binary大小一致
    :return:      path (list of (row, col)) 或 None 表示无法到达
    """
    h, w = binary.shape
    INF = 1e9
    dist = np.full((h, w), INF, dtype=np.float32)
    dist[start[0], start[1]] = 0.0

    parent = {}
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
            if 0 <= ny < h and 0 <= nx < w:
                if binary[ny, nx] == 255:  # 通路
                    # cost = 1/(dt[ny, nx]+1)，离墙越远 dt越大，cost越小
                    step_cost = 1.0 / (dt[ny, nx] + 1.0)
                    new_cost = cur_cost + step_cost
                    if new_cost < dist[ny, nx]:
                        dist[ny, nx] = new_cost
                        parent[(ny, nx)] = (cy, cx)
                        heapq.heappush(pq, (new_cost, (ny, nx)))

    if dist[end[0], end[1]] == INF:
        return None

    # 回溯
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
    简单滑动平均平滑，将原始路径平滑化。
    """
    if window_size < 3:
        return path
    half_win = window_size // 2
    smoothed = []
    for i in range(len(path)):
        window = path[max(0, i - half_win): min(len(path), i + half_win + 1)]
        avg_r = int(round(sum(pt[0] for pt in window) / len(window)))
        avg_c = int(round(sum(pt[1] for pt in window) / len(window)))
        smoothed.append((avg_r, avg_c))
    return smoothed

if __name__ == "__main__":
    image_path = "Puzzle8.png"  # 替换为你的迷宫图文件名
    original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_img is None:
        raise FileNotFoundError("无法读取图像，请检查路径。")

    # ----------------------------
    # 1. 转HSV，提取白色通路 & 黑色线条
    # ----------------------------
    hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)

    # (A) 白色(或浅灰)范围 - 尝试放宽
    # H:0~180都可，S:0~80(饱和度不高)，V:>=180(比较亮)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 80, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # (B) 黑色线条范围 - 放宽
    # H:0~180, S:0~255, V:0~80(比较暗)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # 形态学操作，让黑线稍微变粗
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_black = cv2.dilate(mask_black, kernel, iterations=1)

    # 通路 = 白色 & 非黑线
    mask_black_inv = cv2.bitwise_not(mask_black)
    final_mask = cv2.bitwise_and(mask_white, mask_black_inv)

    # ----------------------------
    # 2. 闭运算，填补外壁裂缝
    # ----------------------------
    # 如果外壁缝隙很小，(5,5) 或 (7,7) 核都可尝试
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # ----------------------------
    # 3. 提取最大轮廓
    # ----------------------------
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise Exception("未找到任何轮廓，请检查颜色范围或形态学参数。")

    max_contour = max(contours, key=cv2.contourArea)
    largest_mask = np.zeros_like(closed_mask)
    cv2.drawContours(largest_mask, [max_contour], -1, 255, thickness=cv2.FILLED)

    final_mask2 = cv2.bitwise_and(closed_mask, largest_mask)

    # ----------------------------
    # 4. 可选：Flood Fill 去除外部
    # ----------------------------
    h, w = final_mask2.shape
    mask_flood = np.zeros((h+2, w+2), np.uint8)
    corners = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
    for (cy, cx) in corners:
        if final_mask2[cy, cx] == 255:
            cv2.floodFill(final_mask2, mask_flood, (cx, cy), 0)

    # ----------------------------
    # 5. 保存/显示中间结果 (调试用)
    # ----------------------------
    # cv2.imwrite("debug_mask_white.png", mask_white)
    # cv2.imwrite("debug_mask_black.png", mask_black)
    # cv2.imwrite("debug_final_mask.png", final_mask)
    # cv2.imwrite("debug_closed_mask.png", closed_mask)
    # cv2.imwrite("debug_largest_mask.png", largest_mask)
    # cv2.imwrite("debug_final_mask2.png", final_mask2)

    # 如果你想在程序中查看，也可用 cv2.imshow(...)
    # cv2.imshow("final_mask2", final_mask2)
    # cv2.waitKey(0)

    # ----------------------------
    # 6. 交互式选择起点终点，必须点击在通路(255)上
    # ----------------------------
    points = []
    display_img = original_img.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            val = final_mask2[y, x]
            print(f"点击({x},{y}), final_mask2像素={val}")
            if val == 255:
                points.append((y, x))
                cv2.circle(display_img, (x, y), 3, (0, 0, 255), -1)
                cv2.imshow("选择起点和终点", display_img)
                if len(points) == 1:
                    print(f"起点已选择: ({x}, {y})")
                elif len(points) == 2:
                    print(f"终点已选择: ({x}, {y})")
            else:
                print("你点击的位置不在通路上，请重新点击！")

    cv2.namedWindow("选择起点和终点", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("选择起点和终点", 800, 600)
    cv2.setMouseCallback("选择起点和终点", mouse_callback)

    print("请在图像上依次点击【起点】和【终点】（都要点在白色通路上），按ESC退出。")
    while len(points) < 2:
        cv2.imshow("选择起点和终点", display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    cv2.destroyWindow("选择起点和终点")

    if len(points) < 2:
        raise Exception("未选择足够的点，请重新运行并选择起点、终点。")

    start_point = points[0]
    end_point = points[1]

    # ----------------------------
    # 7. 距离变换 + Dijkstra
    # ----------------------------
    dist_transform = cv2.distanceTransform(final_mask2, cv2.DIST_L2, 3)
    path = dijkstra_path(final_mask2, start_point, end_point, dist_transform)
    if path is None:
        raise Exception("找不到从起点到终点的路径，可能通路不连通。")

    # ----------------------------
    # 8. 平滑并可视化
    # ----------------------------
    smoothed_path = smooth_path(path, window_size=7)
    solution_img = original_img.copy()
    pts = np.array([[p[1], p[0]] for p in smoothed_path], np.int32).reshape((-1, 1, 2))
    cv2.polylines(solution_img, [pts], isClosed=False, color=(0, 255, 0), thickness=5)

    cv2.namedWindow("最终迷宫路径", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("最终迷宫路径", 800, 600)
    cv2.imshow("最终迷宫路径", solution_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("solved_008.png", solution_img)
    print("结果已保存为 solved_9.png")

