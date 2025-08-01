# import numpy as np
# import cv2
# from collections import deque
# from scipy.interpolate import splprep, splev
#
#
# def find_maze_edges(binary):
#     """查找迷宫的最外围轮廓，并确定出入口"""
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # 选取面积最大的轮廓，假设为迷宫整体
#     largest_contour = max(contours, key=cv2.contourArea)
#
#     # 绘制迷宫掩码
#     h, w = binary.shape
#     maze_mask = np.zeros((h, w), dtype=np.uint8)
#     cv2.drawContours(maze_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
#
#     # 寻找轮廓中靠近图像边界的点作为入口/出口
#     entry_exit = []
#     for point in largest_contour.squeeze():
#         x, y = point
#         if x == 0 or x == w - 1 or y == 0 or y == h - 1:
#             entry_exit.append((y, x))
#
#     # 如果找到的缺口点太多，取第一个和最后一个
#     if len(entry_exit) > 2:
#         entry_exit = [entry_exit[0], entry_exit[-1]]
#
#     # 如果只有一个缺口，则用轮廓中心作为另一个端点
#     if len(entry_exit) == 1:
#         M = cv2.moments(largest_contour)
#         center_x = int(M["m10"] / M["m00"])
#         center_y = int(M["m01"] / M["m00"])
#         entry_exit.append((center_y, center_x))
#
#     return maze_mask, entry_exit
#
#
# def bfs_path(binary, start, end):
#     """在二值图上用 BFS 寻找路径，并避免紧贴墙壁"""
#     h, w = binary.shape
#     dist = np.full((h, w), -1, dtype=np.int32)
#     dist[start[0], start[1]] = 0
#
#     parent = {}
#     queue = deque([start])
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#
#     # 计算每个像素到墙壁的距离
#     dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
#     safe_threshold = 5  # 安全距离，避免贴墙
#
#     while queue:
#         cy, cx = queue.popleft()
#         if (cy, cx) == end:
#             break
#         for dy, dx in directions:
#             ny, nx = cy + dy, cx + dx
#             if 0 <= ny < h and 0 <= nx < w:
#                 if binary[ny, nx] == 255 and dist[ny, nx] == -1:
#                     if dist_transform[ny, nx] > safe_threshold:  # 保证离墙足够远
#                         dist[ny, nx] = dist[cy, cx] + 1
#                         parent[(ny, nx)] = (cy, cx)
#                         queue.append((ny, nx))
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
# def smooth_path(path):
#     """使用样条曲线平滑路径"""
#     path = np.array(path)
#     # 使用样条曲线拟合，s 参数控制平滑程度
#     tck, u = splprep([path[:, 1], path[:, 0]], s=5)
#     u_new = np.linspace(0, 1, num=len(path) * 2)  # 生成更多采样点
#     x_smooth, y_smooth = splev(u_new, tck)
#     smoothed = list(zip(np.rint(y_smooth).astype(int), np.rint(x_smooth).astype(int)))
#     return smoothed
#
#
# def main():
#     # 读取图像
#     image_path = "PuzzleNotWork2.png"
#     original = cv2.imread(image_path)
#     if original is None:
#         raise FileNotFoundError("找不到图片")
#
#     # 转灰度并二值化
#     gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
#
#     # 识别迷宫最外围边缘和出入口
#     maze_mask, entry_exit = find_maze_edges(binary)
#
#     if len(entry_exit) < 2:
#         print("未能找到两个出入口")
#         return
#
#     start, end = entry_exit[0], entry_exit[1]
#     print(f"起点: {start}, 终点: {end}")
#
#     # 用 BFS 寻路
#     path = bfs_path(maze_mask, start, end)
#     if path is None:
#         print("找不到路径")
#         return
#
#     # 对路径进行平滑处理
#     smoothed_path = smooth_path(path)
#
#     # 在原图上绘制路径
#     result = original.copy()
#     for i in range(len(smoothed_path) - 1):
#         y1, x1 = smoothed_path[i]
#         y2, x2 = smoothed_path[i + 1]
#         cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
#
#     # 标记起点和终点
#     cv2.circle(result, (start[1], start[0]), 5, (0, 255, 0), -1)
#     cv2.circle(result, (end[1], end[0]), 5, (255, 0, 0), -1)
#
#     # 显示并保存结果
#     cv2.imshow("迷宫路径", result)
#     cv2.imwrite("maze_solution.png", result)
#     print("已保存结果为 maze_solution.png")
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()
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
#     image_path = "Puzzle10.png"
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
#         cv2.imwrite("maze_solution.png", result)
#         print("已保存结果为 maze_solution.png")
#
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()


import numpy as np
from collections import deque
import cv2

def bfs_path(binary, start, end):
    """在二值图上用BFS寻找路径"""
    h, w = binary.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    dist[start[0], start[1]] = 0

    parent = {}
    queue = deque([start])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        cy, cx = queue.popleft()
        if (cy, cx) == end:
            break
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                if binary[ny, nx] == 255 and dist[ny, nx] == -1:
                    dist[ny, nx] = dist[cy, cx] + 1
                    parent[(ny, nx)] = (cy, cx)
                    queue.append((ny, nx))

    if dist[end[0], end[1]] == -1:
        return None

    path = []
    cur = end
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    path.reverse()
    return path


def on_mouse(event, x, y, flags, param):
    """鼠标回调函数"""
    global points, display_img, maze_paths
    if event == cv2.EVENT_LBUTTONDOWN:
        # 只允许在迷宫通路上选点
        if maze_paths[y, x] == 255:
            points.append((y, x))
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("选择起点和终点", display_img)
            if len(points) == 1:
                print(f"起点: ({y},{x})")
            elif len(points) == 2:
                print(f"终点: ({y},{x})")
        else:
            print("选择的位置不是通路，请重选")


def main():
    global points, display_img, maze_paths

    # 用户可调参数：
    wall_margin = 3     # 离墙距离的安全边界（像素数），可根据需要调整
    balloon_expand = 40   # 气球外扩的距离（像素数），可根据需要调整

    # 读取图像
    image_path = "Puzzle10.png"
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError("找不到图片")

    # 转灰度并二值化
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # =============================================
    # 关键修改：手动创建气球内部掩码
    # =============================================

    # 1. 创建一个空白掩码
    h, w = binary.shape
    balloon_mask = np.zeros((h, w), dtype=np.uint8)

    # 2. 手动绘制气球形状（从图中提取轮廓）
    balloon_center = (w // 2, h // 2)  # 气球中心点
    # 原始半径并外扩 balloon_expand
    balloon_radius = (min(w, h) // 2 - 10) + balloon_expand
    cv2.circle(balloon_mask, balloon_center, balloon_radius, 255, -1)

    # 3. 将气球与迷宫线条叠加
    # 从原始二值图获取迷宫线条（墙壁为黑色）
    maze_walls = cv2.bitwise_not(binary)  # 反转，使墙壁变为白色

    # 加粗墙壁以防止路径穿墙，同时使通路远离墙面
    kernel_size = 2 * wall_margin + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    thick_walls = cv2.dilate(maze_walls, kernel, iterations=1)

    # 从气球掩码中去除墙壁
    maze_paths = cv2.bitwise_and(balloon_mask, cv2.bitwise_not(thick_walls))

    # 保存处理后的图像以便调试
    cv2.imwrite("debug_balloon_mask.png", balloon_mask)
    cv2.imwrite("debug_maze_walls.png", thick_walls)
    cv2.imwrite("debug_maze_paths.png", maze_paths)

    # 交互式选择起点和终点
    points = []
    display_img = original.copy()

    cv2.namedWindow("选择起点和终点", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("选择起点和终点", 800, 800)
    cv2.setMouseCallback("选择起点和终点", on_mouse)

    # 显示处理后的掩码，便于选点
    cv2.namedWindow("迷宫通路", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("迷宫通路", 800, 800)
    cv2.imshow("迷宫通路", maze_paths)

    print("请在迷宫通路（白色区域）上依次点击起点和终点，然后按ESC")

    while True:
        cv2.imshow("选择起点和终点", display_img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or len(points) >= 2:  # ESC或选择了两个点
            break

    if len(points) < 2:
        print("未选择足够的点")
        return

    # 寻路
    path = bfs_path(maze_paths, points[0], points[1])

    if path is None:
        print("找不到路径")
    else:
        print(f"找到路径，长度={len(path)}")

        # 在原图上绘制路径
        result = original.copy()
        for i in range(len(path) - 1):
            y1, x1 = path[i]
            y2, x2 = path[i + 1]
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 标记起点和终点
        cv2.circle(result, (points[0][1], points[0][0]), 5, (0, 255, 0), -1)
        cv2.circle(result, (points[1][1], points[1][0]), 5, (255, 0, 0), -1)

        # 显示结果
        cv2.namedWindow("迷宫路径", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("迷宫路径", 800, 800)
        cv2.imshow("迷宫路径", result)
        cv2.waitKey(0)

        # 保存结果
        cv2.imwrite("maze_solution.png", result)
        print("已保存结果为 maze_solution.png")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
