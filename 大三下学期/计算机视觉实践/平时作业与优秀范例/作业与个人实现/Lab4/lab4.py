# 先导入相关的库，定义显示图像函数
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


        