import cv2
import numpy as np

# 读取两张图片（请根据实际情况修改图片路径）
img1 = cv2.imread("1.jfif")
img2 = cv2.imread("final_result.png")

# 确保两张图片尺寸一致，如果不一致，则统一调整为第一张的尺寸
height, width = img1.shape[:2]
img2 = cv2.resize(img2, (width, height))

# 视频参数
fps = 30
frames_img1 = 2 * fps  # 前两秒
frames_img2 = 3 * fps  # 后三秒

# 初始化视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output_video.avi", fourcc, fps, (width, height))

# 模拟拍照时的抖动参数
# 可调参数：最大旋转角度范围（度）、最大平移像素范围
max_angle = 0.4 # -2到2度
max_translation = 5 # -5到5像素
# 每隔 update_interval 帧更新一次目标抖动参数（平滑过渡降低抖动速度）
update_interval = 5

# 初始化当前抖动参数
current_angle = 0.0
current_tx = 0
current_ty = 0

# 随机生成第一个目标抖动参数
target_angle = np.random.uniform(-max_angle, max_angle)
target_tx = np.random.randint(-max_translation, max_translation + 1)
target_ty = np.random.randint(-max_translation, max_translation + 1)

for i in range(frames_img1):
    # 每隔 update_interval 帧更新目标参数
    if i % update_interval == 0 and i != 0:
        current_angle = target_angle
        current_tx = target_tx
        current_ty = target_ty
        target_angle = np.random.uniform(-max_angle, max_angle)
        target_tx = np.random.randint(-max_translation, max_translation + 1)
        target_ty = np.random.randint(-max_translation, max_translation + 1)
        frame_in_interval = 0
    else:
        frame_in_interval = i % update_interval

    # 线性插值当前帧的抖动参数
    alpha = frame_in_interval / update_interval
    angle = (1 - alpha) * current_angle + alpha * target_angle
    tx = int((1 - alpha) * current_tx + alpha * target_tx)
    ty = int((1 - alpha) * current_ty + alpha * target_ty)

    # 获取旋转矩阵（图像中心为旋转中心）
    M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    M[0, 2] += tx
    M[1, 2] += ty

    # 对 img1 进行仿射变换，生成当前帧
    frame = cv2.warpAffine(img1, M, (width, height))
    out.write(frame)
# 后三秒：直接输出第二张图片
for i in range(frames_img2):
    out.write(img2)

out.release()
print("视频已保存到 output_video.avi")
