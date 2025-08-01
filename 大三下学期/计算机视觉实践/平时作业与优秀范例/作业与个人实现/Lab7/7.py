import numpy as np
import cv2


class Stitcher:
    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # 检查输入是否为两张图片
        if images is None or len(images) != 2:
            print("Error: 输入图片数量不正确。")
            return None

        (imageA, imageB) = images

        # 检查图片是否加载成功
        if imageA is None or imageB is None:
            print("Error: 无法加载图片，请检查文件路径！")
            return None

        # 检测两张图片的 SIFT 关键点与描述子
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # 判断是否检测到足够关键点
        if featuresA is None or featuresB is None or len(kpsA) == 0 or len(kpsB) == 0:
            print("Error: 一张或多张图片中未检测到足够的特征。")
            return None

        # 匹配两张图片的特征点
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        if M is None:
            print("Error: 匹配关键点不足！")
            return None

        # 提取匹配结果和变换矩阵 H
        (matches, H, status) = M

        # 对图片A进行透视变换，结果图大小设置为 imageA 与 imageB 宽度相加，高度为 imageA 高度
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        # 将 imageB 贴到结果图左侧
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # 如需显示匹配结果，则绘制匹配连线
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            return (result, vis)

        return result

    def detectAndDescribe(self, image):
        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 创建 SIFT 检测器
        sift = cv2.SIFT_create()
        # 检测关键点和计算描述子，使用灰度图像提升一致性
        (kps, features) = sift.detectAndCompute(gray, None)
        if kps is None or len(kps) == 0:
            print("Warning: 图片中未检测到 SIFT 关键点。")
            return (np.array([]), None)
        # 将关键点坐标转换为 NumPy 数组
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # 检查描述子是否存在
        if featuresA is None or featuresB is None or len(featuresA) == 0 or len(featuresB) == 0:
            print("Error: 特征描述子为空。")
            return None

        # 建立暴力匹配器
        matcher = cv2.BFMatcher()
        # 使用 KNN 方法（K=2）查找匹配
        rawMatches = matcher.knnMatch(featuresA, featuresB, k=2)
        matches = []
        # 筛选有效匹配：最近邻比值小于指定阈值
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                # 存储匹配对，按照 (trainIdx, queryIdx) 的顺序
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # 至少需要四组匹配对来计算透视矩阵
        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # 计算透视变换矩阵 H，同时返回匹配状态
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            return (matches, H, status)
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 创建一幅新图，将 imageA 与 imageB 横向拼接
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:wA + wB] = imageB

        # 遍历所有匹配，画出匹配连线
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:  # 仅画出状态为 1 的有效匹配
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        return vis


if __name__ == '__main__':
    # 读取图片
    imageA = cv2.imread("1.JPG")
    imageB = cv2.imread("2.JPG")

    if imageA is None or imageB is None:
        print("Error: 无法加载图片，请检查图片路径。")
        exit()

    stitcher = Stitcher()
    # 若需要显示匹配连线，则设置 showMatches=True
    result = stitcher.stitch([imageA, imageB], showMatches=False)

    if result is None:
        print("图片拼接失败！")
    else:
        # cv2.imshow("Stitched Result", result)
        cv2.imwrite("result.jpg", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
