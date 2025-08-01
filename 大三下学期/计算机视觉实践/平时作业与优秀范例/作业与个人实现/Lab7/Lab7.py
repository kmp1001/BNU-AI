# import cv2
# import numpy as np
# import glob
# import os
#
#
# ####################################
# # 1. 多频带融合相关函数实现
# ####################################
#
# def build_gaussian_pyramid(img, levels):
#     """构建高斯金字塔"""
#     gp = [img.astype(np.float32)]
#     for i in range(levels):
#         img = cv2.pyrDown(img)
#         gp.append(img.astype(np.float32))
#     return gp
#
#
# def build_laplacian_pyramid(gp):
#     """由高斯金字塔构建拉普拉斯金字塔"""
#     lp = []
#     for i in range(len(gp) - 1):
#         size = (gp[i].shape[1], gp[i].shape[0])
#         up = cv2.pyrUp(gp[i + 1], dstsize=size)
#         lap = cv2.subtract(gp[i], up)
#         lp.append(lap)
#     lp.append(gp[-1])
#     return lp
#
#
# def multi_band_blend(img1, img2, mask, num_levels=5):
#     """
#     使用多频带融合（Burt–Adelson 算法）融合两幅图像
#     融合公式： result = img1*mask + img2*(1-mask)
#     参数 mask 为单通道图，取值范围 [0, 255]（内部会归一化）
#     """
#     # 转为 float32
#     img1 = img1.astype(np.float32)
#     img2 = img2.astype(np.float32)
#     if len(mask.shape) != 2:
#         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#     mask = mask.astype(np.float32) / 255.0
#
#     gp_img1 = build_gaussian_pyramid(img1, num_levels)
#     gp_img2 = build_gaussian_pyramid(img2, num_levels)
#     gp_mask = build_gaussian_pyramid(mask, num_levels)
#
#     lp_img1 = build_laplacian_pyramid(gp_img1)
#     lp_img2 = build_laplacian_pyramid(gp_img2)
#
#     blended_pyramid = []
#     for L1, L2, gm in zip(lp_img1, lp_img2, gp_mask):
#         # 对于彩色图像，将 gm 扩展到3通道
#         gm_3 = cv2.merge([gm, gm, gm])
#         blended = L1 * gm_3 + L2 * (1 - gm_3)
#         blended_pyramid.append(blended)
#
#     blended_img = blended_pyramid[-1]
#     for i in range(num_levels - 1, -1, -1):
#         size = (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0])
#         blended_img = cv2.pyrUp(blended_img, dstsize=size)
#         blended_img = cv2.add(blended_img, blended_pyramid[i])
#
#     blended_img = np.clip(blended_img, 0, 255).astype(np.uint8)
#     return blended_img
#
#
# ####################################
# # 2. 裁剪黑边函数：提取有效区域
# ####################################
# def crop_black_border(image):
#     """
#     裁剪图像四周的黑边（假定背景为纯黑）
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # 二值化，非零区域为有效区域
#     _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return image
#     x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
#     return image[y:y + h, x:x + w]
#
#
# ####################################
# # 3. 迭代拼接函数：两幅图像配准、变换、融合
# ####################################
# def stitch_two_images_cpu(img1, img2):
#     """
#     使用 ORB 特征检测与 BFMatcher 进行图像配准；
#     计算 img2 到 img1 的单应性矩阵（使用 RANSAC）；
#     透视变换 img2 到 img1 坐标系，并利用多频带融合融合两图。
#     返回拼接后的结果图，若匹配失败则返回 None。
#     """
#     # ① 特征检测及描述
#     orb = cv2.ORB_create(nfeatures=3000)
#     kp1, des1 = orb.detectAndCompute(img1, None)
#     kp2, des2 = orb.detectAndCompute(img2, None)
#
#     if des1 is None or des2 is None:
#         print("描述子为空，配准失败")
#         return None
#
#     # ② 特征匹配
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING)
#     knn_matches = bf.knnMatch(des1, des2, k=2)
#     good_matches = []
#     for m, n in knn_matches:
#         if m.distance < 0.8 * n.distance:  # 可根据需要调整阈值
#             good_matches.append(m)
#     print(f"匹配到的好匹配数量: {len(good_matches)}")
#     if len(good_matches) < 4:
#         print("匹配点不足，无法计算单应性矩阵")
#         return None
#
#     # ③ 构造匹配点对
#     pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#     pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#     # ④ 求解单应性矩阵
#     H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
#     if H is None:
#         print("无法计算单应性矩阵")
#         return None
#
#     # ⑤ 根据 H 计算拼接后画布尺寸
#     h1, w1 = img1.shape[:2]
#     h2, w2 = img2.shape[:2]
#     corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
#     warped_corners_img2 = cv2.perspectiveTransform(corners_img2, H)
#     corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
#     all_corners = np.concatenate((corners_img1, warped_corners_img2), axis=0)
#     [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
#     [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
#
#     # ⑥ 计算平移补偿矩阵
#     translation = [-xmin, -ymin]
#     H_translation = np.array([[1, 0, translation[0]],
#                               [0, 1, translation[1]],
#                               [0, 0, 1]])
#
#     canvas_width = xmax - xmin
#     canvas_height = ymax - ymin
#
#     # ⑦ 对 img2 进行透视变换
#     warped_img2 = cv2.warpPerspective(img2, H_translation.dot(H), (canvas_width, canvas_height))
#     # ⑧ 将 img1 放置到对应位置
#     canvas_img1 = np.zeros((canvas_height, canvas_width, 3), dtype=img1.dtype)
#     canvas_img1[translation[1]:translation[1] + h1, translation[0]:translation[0] + w1] = img1
#
#     # ⑨ 构造重叠区域权重图：简单采用线性权重
#     # 这里根据 canvas_img1 中已有像素生成掩码
#     mask1 = cv2.cvtColor(canvas_img1, cv2.COLOR_BGR2GRAY)
#     mask1 = (mask1 > 0).astype(np.uint8) * 255
#     mask2 = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY)
#     mask2 = (mask2 > 0).astype(np.uint8) * 255
#     sum_mask = mask1.astype(np.float32) + mask2.astype(np.float32) + 1e-6
#     alpha = (mask1.astype(np.float32) / sum_mask)
#     alpha = (alpha * 255).astype(np.uint8)
#
#     # ⑩ 利用多频带融合对 canvas_img1 与 warped_img2 进行融合
#     blended = multi_band_blend(canvas_img1, warped_img2, alpha, num_levels=5)
#
#     return blended
#
#
# ####################################
# # 4. 迭代拼接多幅图像（自实现，不调用 Stitcher）
# ####################################
# def manual_iterative_stitching(images):
#     """
#     以第一张图像作为初始全景图，
#     依次调用 stitch_two_images_cpu 与 crop_black_border 进行迭代拼接。
#     如果某次匹配失败，则跳过该图像。
#     """
#     if not images:
#         print("未输入图像列表")
#         return None
#     panorama = images[0]
#     for i in range(1, len(images)):
#         print(f"正在拼接第 {i + 1} 张图像...")
#         result = stitch_two_images_cpu(panorama, images[i])
#         if result is None:
#             print(f"第 {i + 1} 张图像匹配失败，跳过该图像")
#             continue
#         panorama = crop_black_border(result)
#     return panorama
#
#
# ####################################
# # 5. 主函数入口：读取图像并进行拼接
# ####################################
# def main():
#     # 修改 image_folder 为实际图片目录
#     image_folder = r"D:\semester 6\CVparctice\Lab7\CV07"  # 请修改为实际目录
#     # 读取所有 jpg 文件，支持大写或小写扩展名
#     image_paths = glob.glob(os.path.join(image_folder, "*.JPG"))
#     image_paths.sort()
#     if len(image_paths) < 2:
#         print("至少需要两张图像进行拼接")
#         return
#
#     print("读取到的图像路径：")
#     for p in image_paths:
#         print(p)
#
#     images = []
#     for p in image_paths:
#         img = cv2.imread(p)
#         if img is None:
#             print(f"读取 {p} 失败！")
#             continue
#         images.append(img)
#
#     # 调用迭代拼接
#     print("开始迭代拼接...")
#     panorama = manual_iterative_stitching(images)
#     if panorama is not None:
#         output_file = "final_panorama.jpg"
#         cv2.imwrite(output_file, panorama)
#         cv2.imshow("最终拼接全景", panorama)
#         print("最终全景图已保存为：", output_file)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     else:
#         print("全景拼接失败")
#
#
# if __name__ == "__main__":
#     main()


import cv2
import numpy as np


def calc_corners(H, src):
    """
    利用单应性矩阵 H 计算 src 图像的四个角在目标图像中的坐标，
    返回字典，包括 'left_top', 'left_bottom', 'right_top', 'right_bottom'
    """
    h, w = src.shape[:2]
    # 定义四个角点的齐次坐标
    pts = np.array([[0, 0, 1],
                    [0, h, 1],
                    [w, 0, 1],
                    [w, h, 1]]).T
    pts_transformed = H.dot(pts)
    pts_transformed /= pts_transformed[2, :]
    corners = {
        'left_top': (pts_transformed[0, 0], pts_transformed[1, 0]),
        'left_bottom': (pts_transformed[0, 1], pts_transformed[1, 1]),
        'right_top': (pts_transformed[0, 2], pts_transformed[1, 2]),
        'right_bottom': (pts_transformed[0, 3], pts_transformed[1, 3])
    }
    return corners


def optimize_seam(img_left, warped_right, dst, start):
    """
    对左图 img_left 与透视变换后的右图 warped_right 的重叠区域进行简单线性融合，
    参数 start 指定重叠区域的起始列（从左图的某列开始进入重叠区）。
    结果写入 dst 图像中并返回。
    """
    rows, cols, _ = img_left.shape
    processWidth = cols - start
    # 遍历重叠区域，按距离重叠左边界的距离生成线性权重
    for i in range(rows):
        for j in range(start, cols):
            # 如果 warped_right 中像素均为 0，则认为仅使用左图
            if np.all(warped_right[i, j] == 0):
                alpha = 1.0
            else:
                alpha = (processWidth - (j - start)) / processWidth
            # 计算融合后的像素（保证数据类型为 uint8）
            dst[i, j] = np.clip(img_left[i, j] * alpha + warped_right[i, j] * (1 - alpha), 0, 255)
    return dst


def main():
    # 读取两幅图像（请确保 t1.jpg 和 t2.jpg 路径正确）
    # 假设 t2.jpg 为左图，t1.jpg 为右图（与 C++ 示例对应）
    img_left = cv2.imread("1.JPG", cv2.IMREAD_COLOR)
    img_right = cv2.imread("2.JPG", cv2.IMREAD_COLOR)

    if img_left is None or img_right is None:
        print("读取图片失败，请检查图片路径")
        return

    cv2.imshow("Left Image", img_left)
    cv2.imshow("Right Image", img_right)
    cv2.waitKey(0)

    # 将图像转换为灰度
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 使用 SIFT 特征检测与描述 (需要 opencv-contrib-python)
    sift = cv2.SIFT_create()
    kp_left, des_left = sift.detectAndCompute(gray_left, None)
    kp_right, des_right = sift.detectAndCompute(gray_right, None)

    # 使用 BFMatcher 进行特征匹配（SIFT 使用 L2 距离）
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_left, des_right, k=2)

    # Lowe's ratio test 筛选
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    print("Good matches count:", len(good_matches))
    if len(good_matches) < 4:
        print("匹配点不足，无法拼接！")
        return

    # 可视化匹配结果
    img_matches = cv2.drawMatches(img_left, kp_left, img_right, kp_right, good_matches, None, flags=2)
    cv2.imshow("Matches", img_matches)
    cv2.waitKey(0)

    # 提取匹配点坐标，注意这里匹配点为左图 query 和右图 train
    pts_left = np.float32([kp_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵：将右图映射到左图坐标系
    H, mask = cv2.findHomography(pts_right, pts_left, cv2.RANSAC, 5.0)
    if H is None:
        print("单应性矩阵计算失败")
        return
    print("Homography Matrix:\n", H)

    # 计算右图经过 H 变换后的四个角
    corners = calc_corners(H, img_right)
    print("Transformed corners:", corners)

    # 确定拼接画布大小：
    # 这里简单采用左图宽度为基础，右图变换后可能超出左图范围
    h_left, w_left = img_left.shape[:2]
    canvas_width = int(max(w_left, corners['right_top'][0], corners['right_bottom'][0]))
    canvas_height = h_left  # 这里假设高度不变

    # 创建空白画布，并将左图拷贝到画布中
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[0:h_left, 0:w_left] = img_left

    # 将右图透视变换到画布上
    warped_right = cv2.warpPerspective(img_right, H, (canvas_width, canvas_height))
    cv2.imshow("Warped Right", warped_right)
    cv2.waitKey(0)

    # 融合：这里先将 warped_right 与左图 canvas 融合为初步结果
    # 这里采用线性融合，在重叠区域使用 optimize_seam 进行平滑处理
    blended = canvas.copy()
    # 这里认为重叠区域在左图从某列开始，例如从 left 图宽度的 60% 开始
    overlap_start = int(w_left * 0.6)
    blended = optimize_seam(img_left, warped_right, canvas.copy(), overlap_start)

    cv2.imshow("Final Blended Panorama", blended)
    cv2.imwrite("dst.jpg", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
