import sys
import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

# ========================= 图像处理类 =========================
class ImageProcessor:
    """
    图像处理类，所有输入图均假设为 RGB 格式 (H, W, 3)。
    除图像读写外，其余全部使用 numpy 自写实现，包括二维DFT/IDFT、移频、频域滤波等。
    """
    # ---------- 颜色转换 ----------
    @staticmethod
    def rgb_to_gray(img_rgb: np.ndarray) -> np.ndarray:
        """
        将 RGB 图像转换为伪灰度图（每个通道均为 0.299*R+0.587*G+0.114*B），结果仍为 3 通道。
        """
        if len(img_rgb.shape) != 3 or img_rgb.shape[2] != 3:
            raise ValueError("输入图像必须为 RGB 格式 (H, W, 3)！")
        R = img_rgb[:, :, 0].astype(np.float32)
        G = img_rgb[:, :, 1].astype(np.float32)
        B = img_rgb[:, :, 2].astype(np.float32)
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return np.stack((gray, gray, gray), axis=-1)

    @staticmethod
    def gray_to_rgb(img_gray: np.ndarray) -> np.ndarray:
        """
        将灰度图转换为 RGB 格式：若单通道则扩展为3通道；若3通道且各通道一致则直接返回，
        否则提示错误（仅针对伪灰度）。
        """
        if len(img_gray.shape) == 2:
            return np.stack((img_gray, img_gray, img_gray), axis=-1)
        elif len(img_gray.shape) == 3 and img_gray.shape[2] == 3:
            if np.allclose(img_gray[:, :, 0], img_gray[:, :, 1]) and np.allclose(img_gray[:, :, 0], img_gray[:, :, 2]):
                return img_gray
            else:
                QtWidgets.QMessageBox.information(None, "提示", "图像并非灰度图，无法转换为彩色！")
                return img_gray
        else:
            raise ValueError("图像格式不支持！")

    # ---------- 空域滤波 ----------
    @staticmethod
    def mean_filter(img: np.ndarray, **kwargs) -> np.ndarray:
        """
        均值滤波（空域）：采用向量化方式实现，避免使用 Python 嵌套循环。
        参数：
          Kernel Size: 正奇数（例如：3, 5, 7,...）
        """
        ksize = kwargs.get("Kernel Size", 3)
        if ksize < 1 or ksize % 2 == 0:
            raise ValueError("核大小必须为正奇数！")
        pad = ksize // 2
        h, w, c = img.shape
        result = np.empty((h, w, c), dtype=np.float64)
        kernel = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)
        for ch in range(c):
            channel = img[:, :, ch]
            padded = np.pad(channel, pad, mode='reflect')
            # 利用滑动窗口构造局部区域数组，窗口 shape 为 (h, w, ksize, ksize)
            windows = np.lib.stride_tricks.sliding_window_view(padded, (ksize, ksize))
            result[:, :, ch] = np.sum(windows * kernel, axis=(-1, -2))
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def gaussian_filter(img: np.ndarray, **kwargs) -> np.ndarray:
        """
        高斯滤波（空域）：采用向量化方式实现滤波，避免双层循环。
        参数：
          Kernel Size: 正奇数
          Sigma: 标准差（浮点数）
        """
        ksize = kwargs.get("Kernel Size", 3)
        sigma = kwargs.get("Sigma", 1.0)
        if ksize < 1 or ksize % 2 == 0:
            raise ValueError("核大小必须为正奇数！")
        pad = ksize // 2
        h, w, c = img.shape
        result = np.empty((h, w, c), dtype=np.float64)
        # 生成高斯核
        ax = np.arange(-pad, pad + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel /= np.sum(kernel)
        for ch in range(c):
            channel = img[:, :, ch]
            padded = np.pad(channel, pad, mode='reflect')
            windows = np.lib.stride_tricks.sliding_window_view(padded, (ksize, ksize))
            result[:, :, ch] = np.sum(windows * kernel, axis=(-1, -2))
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def hist_equalize(img: np.ndarray, **kwargs) -> np.ndarray:
        """
        对 RGB 图像各通道分别进行直方图均衡化。
        """
        h, w, c = img.shape
        if c != 3:
            raise ValueError("输入图像必须为 RGB 格式！")
        out = np.zeros_like(img)
        for ch in range(3):
            out[:, :, ch] = ImageProcessor.hist_equalize_single(img[:, :, ch])
        return out

    @staticmethod
    def hist_equalize_single(channel: np.ndarray) -> np.ndarray:
        """
        对单通道图像做直方图均衡化。
        """
        hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255.0 / cdf[-1]
        eq_channel = np.interp(channel.flatten(), bins[:-1], cdf_normalized)
        return eq_channel.reshape(channel.shape).astype(np.uint8)

    @staticmethod
    def compute_histogram(img: np.ndarray):
        """
        计算 RGB 图像各通道直方图，返回 [R_hist, G_hist, B_hist]。
        """
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError("输入图像必须为 RGB 格式！")
        hists = []
        for ch in range(3):
            hist, _ = np.histogram(img[:, :, ch].flatten(), 256, [0, 256])
            hists.append(hist)
        return hists

    # ---------- 手写频域变换与移频 ----------
    @staticmethod
    def dft_2d(img: np.ndarray) -> np.ndarray:
        """
        手写二维离散傅里叶变换 (DFT)。
        仅适用于小尺寸图像（计算量巨大）。
        """
        M, N = img.shape
        n = np.arange(N)
        k = n.reshape((N, 1))
        W_N = np.exp(-2j * np.pi * k * n / N)
        X_row = np.dot(img, W_N.T)
        m = np.arange(M)
        l = m.reshape((M, 1))
        W_M = np.exp(-2j * np.pi * l * m / M)
        X = np.dot(W_M, X_row)
        return X

    @staticmethod
    def idft_2d(freq: np.ndarray) -> np.ndarray:
        """
        手写二维逆离散傅里叶变换 (IDFT)。
        """
        M, N = freq.shape
        m = np.arange(M)
        l = m.reshape((M, 1))
        W_M_inv = np.exp(2j * np.pi * l * m / M)
        n = np.arange(N)
        k = n.reshape((N, 1))
        W_N_inv = np.exp(2j * np.pi * k * n / N)
        x_row = np.dot(W_M_inv, freq)
        x = np.dot(x_row, W_N_inv.T)
        return x / (M * N)

    @staticmethod
    def fftshift_2d(x: np.ndarray) -> np.ndarray:
        """
        手写二维移频，将零频分量移至频谱中心。
        """
        M, N = x.shape
        m2 = (M + 1) // 2
        n2 = (N + 1) // 2
        top = np.hstack((x[m2:, n2:], x[m2:, :n2]))
        bottom = np.hstack((x[:m2, n2:], x[:m2, :n2]))
        return np.vstack((top, bottom))

    @staticmethod
    def ifftshift_2d(x: np.ndarray) -> np.ndarray:
        """
        手写二维反移频，将中心零频移回左上角。
        """
        M, N = x.shape
        m2 = M - (M + 1) // 2
        n2 = N - (N + 1) // 2
        top = np.hstack((x[m2:, n2:], x[m2:, :n2]))
        bottom = np.hstack((x[:m2, n2:], x[:m2, :n2]))
        return np.vstack((top, bottom))

    # ---------- 频域滤波 ----------
    @staticmethod
    def butterworth_lowpass_filter(img: np.ndarray, **kwargs) -> np.ndarray:
        """
        巴特沃斯低通滤波（频域处理）：对 RGB 图像各通道分别处理。
        参数：
          D0: 截止频率
          Order n: 阶数
        """
        channels = cv2.split(img)
        filtered_channels = []
        for ch in channels:
            filtered_channels.append(ImageProcessor._butterworth_lowpass_single(
                ch, kwargs.get("D0", 30), kwargs.get("Order n", 2)))
        return cv2.merge(filtered_channels)

    @staticmethod
    def _butterworth_lowpass_single(channel: np.ndarray, d0=30, n=2) -> np.ndarray:
        f = ImageProcessor.dft_2d(channel.astype(np.float32))
        fshift = ImageProcessor.fftshift_2d(f)
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        U, V = np.meshgrid(np.arange(cols), np.arange(rows))
        D_uv = np.sqrt((V - crow) ** 2 + (U - ccol) ** 2)
        H = 1 / (1 + (D_uv / d0) ** (2 * n))
        fshift_filtered = fshift * H
        f_ishift = ImageProcessor.ifftshift_2d(fshift_filtered)
        img_back = ImageProcessor.idft_2d(f_ishift)
        img_back = np.real(img_back)
        return np.clip(img_back, 0, 255).astype(np.uint8)

    @staticmethod
    def gaussian_highpass_filter(img: np.ndarray, **kwargs) -> np.ndarray:
        """
        高斯高通滤波（频域处理）：对 RGB 各通道分别处理后合并。
        参数：
          D0: 截止频率
        """
        channels = cv2.split(img)
        filtered_channels = []
        for ch in channels:
            filtered_channels.append(ImageProcessor._gaussian_highpass_single(
                ch, kwargs.get("D0", 30)))
        return cv2.merge(filtered_channels)

    @staticmethod
    def _gaussian_highpass_single(channel: np.ndarray, d0=30) -> np.ndarray:
        f = ImageProcessor.dft_2d(channel.astype(np.float32))
        fshift = ImageProcessor.fftshift_2d(f)
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        U, V = np.meshgrid(np.arange(cols), np.arange(rows))
        D_uv = np.sqrt((V - crow) ** 2 + (U - ccol) ** 2)
        lowpass = np.exp(- (D_uv ** 2) / (2 * (d0 ** 2)))
        highpass = 1 - lowpass
        fshift_filtered = fshift * highpass
        f_ishift = ImageProcessor.ifftshift_2d(fshift_filtered)
        img_back = ImageProcessor.idft_2d(f_ishift)
        img_back = np.real(img_back)
        return np.clip(img_back, 0, 255).astype(np.uint8)

# ========================= 参数对话框（含实验原理说明） =========================
class ParameterDialog(QtWidgets.QDialog):
    """
    通用参数对话框，支持滑动条、数字显示及实时预览。
    使用 QTabWidget 分为两页：
      - "参数调节" 页：包含预览区域与参数控件，实时显示不同参数对图像的影响；
      - "实验原理" 页：以文字说明展示该实验（滤波）原理及效果。
    直接传入参数配置列表、原图和处理函数（要求支持关键字参数）。
    """
    def __init__(self, parent, title, original_image, transform_function, params, explanation=""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.original_image = original_image.copy()
        self.transform_function = transform_function
        self.params_config = params  # 参数配置列表
        self.controls = {}  # 存放各参数控件
        self.explanation_text = explanation
        self.initUI()

    def initUI(self):
        self.setSizeGripEnabled(True)
        tabWidget = QtWidgets.QTabWidget(self)

        # Tab1: 参数调节与预览
        tab1 = QtWidgets.QWidget()
        vlayout1 = QtWidgets.QVBoxLayout(tab1)

        self.previewLabel = QtWidgets.QLabel("预览区域")
        self.previewLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.previewLabel.setMinimumHeight(300)
        vlayout1.addWidget(self.previewLabel)

        # 参数控件区域
        for config in self.params_config:
            hlayout = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(config["name"] + ":")
            hlayout.addWidget(label)
            if config["type"] == "int":
                spin = QtWidgets.QSpinBox()
                spin.setMinimum(config["min"])
                spin.setMaximum(config["max"])
                spin.setSingleStep(config.get("step", 1))
                spin.setValue(config["default"])
                if config.get("odd", False) and spin.value() % 2 == 0:
                    spin.setValue(spin.value() + 1)
                spin.valueChanged.connect(self.updatePreview)
                hlayout.addWidget(spin)
                self.controls[config["name"]] = spin
            elif config["type"] == "float":
                dspin = QtWidgets.QDoubleSpinBox()
                dspin.setMinimum(config["min"])
                dspin.setMaximum(config["max"])
                dspin.setSingleStep(config.get("step", 0.1))
                dspin.setValue(config["default"])
                dspin.valueChanged.connect(self.updatePreview)
                hlayout.addWidget(dspin)
                self.controls[config["name"]] = dspin
            vlayout1.addLayout(hlayout)

        # 操作按钮区域
        btnLayout = QtWidgets.QHBoxLayout()
        self.applyBtn = QtWidgets.QPushButton("应用")
        self.applyBtn.clicked.connect(self.accept)
        self.resetBtn = QtWidgets.QPushButton("重置")
        self.resetBtn.clicked.connect(self.resetParameters)
        self.cancelBtn = QtWidgets.QPushButton("取消")
        self.cancelBtn.clicked.connect(self.reject)
        btnLayout.addWidget(self.applyBtn)
        btnLayout.addWidget(self.resetBtn)
        btnLayout.addWidget(self.cancelBtn)
        vlayout1.addLayout(btnLayout)

        # Tab2: 实验原理说明
        tab2 = QtWidgets.QWidget()
        vlayout2 = QtWidgets.QVBoxLayout(tab2)
        explanationEdit = QtWidgets.QPlainTextEdit()
        explanationEdit.setPlainText(self.explanation_text)
        explanationEdit.setReadOnly(True)
        explanationEdit.setWordWrapMode(QtGui.QTextOption.WordWrap)
        vlayout2.addWidget(explanationEdit)

        tabWidget.addTab(tab1, "参数调节")
        tabWidget.addTab(tab2, "实验原理")

        mainLayout = QtWidgets.QVBoxLayout(self)
        mainLayout.addWidget(tabWidget)

        self.updatePreview()

    def resetParameters(self):
        for config in self.params_config:
            widget = self.controls[config["name"]]
            widget.setValue(config["default"])
        self.updatePreview()

    def getParameters(self):
        params = {}
        for config in self.params_config:
            widget = self.controls[config["name"]]
            params[config["name"]] = widget.value()
        return params

    def updatePreview(self):
        params = self.getParameters()
        try:
            result = self.transform_function(self.original_image, **params)
            h, w, ch = result.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(result.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimg)
            self.previewLabel.setPixmap(pixmap.scaled(self.previewLabel.size(),
                                                      QtCore.Qt.KeepAspectRatio,
                                                      QtCore.Qt.SmoothTransformation))
        except Exception as e:
            self.previewLabel.setText("预览失败: " + str(e))

    def getResultImage(self):
        params = self.getParameters()
        return self.transform_function(self.original_image, **params)

# ========================= 直方图对话框 =========================
class HistogramDialog(QtWidgets.QDialog):
    """
    直方图显示对话框，使用 QPainter 绘制 RGB 三通道直方图，并能随着对话框尺寸调整自适应重绘。
    """
    def __init__(self, parent, image, title="直方图"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.image = image
        self.initUI()
        self.setSizeGripEnabled(True)

    def initUI(self):
        layout = QtWidgets.QVBoxLayout(self)
        self.histLabel = QtWidgets.QLabel()
        self.histLabel.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.histLabel)
        self.drawHistogram()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.drawHistogram()  # 每次尺寸变化时重绘直方图

    def drawHistogram(self):
        hists = ImageProcessor.compute_histogram(self.image)  # [R, G, B]
        # 动态获取绘图区域尺寸，若尺寸为0则使用默认值
        width = self.histLabel.width() if self.histLabel.width() > 0 else 512
        height = self.histLabel.height() if self.histLabel.height() > 0 else 400
        pixmap = QtGui.QPixmap(width, height)
        pixmap.fill(QtCore.Qt.white)
        painter = QtGui.QPainter(pixmap)
        # 绘制坐标轴
        pen = QtGui.QPen(QtCore.Qt.black)
        painter.setPen(pen)
        painter.drawLine(50, height - 30, width - 20, height - 30)  # X轴
        painter.drawLine(50, 20, 50, height - 30)  # Y轴
        max_val = max(max(hists[0]), max(hists[1]), max(hists[2]))
        scale = (height - 50) / max_val if max_val > 0 else 1
        bin_width = (width - 70) / 256.0
        colors = [QtCore.Qt.red, QtCore.Qt.green, QtCore.Qt.blue]
        for idx, hist in enumerate(hists):
            pen = QtGui.QPen(colors[idx])
            painter.setPen(pen)
            points = []
            for i in range(256):
                x = 50 + i * bin_width
                y = height - 30 - hist[i] * scale
                points.append(QtCore.QPointF(x, y))
            painter.drawPolyline(QtGui.QPolygonF(points))
        painter.end()
        self.histLabel.setPixmap(pixmap)

# ========================= 主窗口 =========================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("综合图像处理示例")
        self.resize(1000, 600)
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            QMenuBar { background-color: #444; color: white; font: bold 14px; }
            QMenuBar::item { background-color: #444; padding: 8px 14px; }
            QMenuBar::item:selected { background-color: #666; }
            QToolBar { background-color: #ddd; }
            QPushButton { background-color: #3498db; color: white; border-radius: 5px; padding: 5px 10px; }
            QSlider::handle:horizontal { background: #3498db; }
        """)
        self.undoStack = []  # 用于撤销操作
        self.img = None     # 当前图像（RGB 格式）
        self._initUI()

    def _initUI(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        self.mainLayout = QtWidgets.QVBoxLayout(central)
        self.imageLabel = QtWidgets.QLabel("在此显示图像")
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setStyleSheet("background-color: #fff; border: 1px solid #ccc;")
        self.mainLayout.addWidget(self.imageLabel)
        self._createMenuBar()
        self._createToolBar()

    def _createMenuBar(self):
        menuBar = self.menuBar()
        # 文件菜单
        fileMenu = menuBar.addMenu("文件")
        openAct = QtWidgets.QAction("导入图片", self)
        openAct.triggered.connect(self.open_image)
        fileMenu.addAction(openAct)
        saveAct = QtWidgets.QAction("保存图片", self)
        saveAct.triggered.connect(self.save_image)
        fileMenu.addAction(saveAct)
        # 色彩变换菜单
        colorMenu = menuBar.addMenu("色彩变换")
        rgb2grayAct = QtWidgets.QAction("RGB转灰度", self)
        rgb2grayAct.triggered.connect(self.action_rgb2gray)
        colorMenu.addAction(rgb2grayAct)
        gray2rgbAct = QtWidgets.QAction("灰度转RGB", self)
        gray2rgbAct.triggered.connect(self.action_gray2rgb)
        colorMenu.addAction(gray2rgbAct)
        # 空域滤波菜单
        spatialMenu = menuBar.addMenu("空域滤波")
        meanFilterAct = QtWidgets.QAction("均值滤波", self)
        meanFilterAct.triggered.connect(self.action_mean_filter)
        spatialMenu.addAction(meanFilterAct)
        gaussFilterAct = QtWidgets.QAction("高斯滤波", self)
        gaussFilterAct.triggered.connect(self.action_gauss_filter)
        spatialMenu.addAction(gaussFilterAct)
        histEqAct = QtWidgets.QAction("直方图均衡化", self)
        histEqAct.triggered.connect(self.action_hist_eq)
        spatialMenu.addAction(histEqAct)
        # 频域滤波菜单
        freqMenu = menuBar.addMenu("频域滤波")
        butterAct = QtWidgets.QAction("巴特沃斯低通滤波", self)
        butterAct.triggered.connect(self.action_butterworth_lowpass)
        freqMenu.addAction(butterAct)
        gaussHpAct = QtWidgets.QAction("高斯高通滤波", self)
        gaussHpAct.triggered.connect(self.action_gaussian_highpass)
        freqMenu.addAction(gaussHpAct)
        # 直方图显示菜单
        histMenu = menuBar.addMenu("图像直方图")
        computeHistAct = QtWidgets.QAction("直方图显示", self)
        computeHistAct.triggered.connect(self.action_compute_histogram)
        histMenu.addAction(computeHistAct)

    def _createToolBar(self):
        toolBar = QtWidgets.QToolBar()
        self.addToolBar(toolBar)
        undoAct = QtWidgets.QAction("撤销", self)
        undoAct.triggered.connect(self.undo)
        toolBar.addAction(undoAct)

    # ---------------- 图像读写 ----------------
    def open_image(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图像", "", "Images (*.png *.jpg *.bmp)")
        if fname:
            img_bgr = cv2.imread(fname)
            if img_bgr is None:
                return
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            self.img = img_rgb
            self.undoStack.clear()
            self.show_image(self.img)

    def save_image(self):
        if self.img is None:
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存图像", "", "Images (*.png *.jpg *.bmp)")
        if fname:
            img_bgr = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(fname, img_bgr)

    def show_image(self, img):
        if img is None:
            return
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.size(),
                                                  QtCore.Qt.KeepAspectRatio,
                                                  QtCore.Qt.SmoothTransformation))

    def push_undo(self):
        if self.img is not None:
            self.undoStack.append(self.img.copy())

    def undo(self):
        if self.undoStack:
            self.img = self.undoStack.pop()
            self.show_image(self.img)
        else:
            QtWidgets.QMessageBox.information(self, "提示", "没有可撤销的操作。")

    def offer_save_image(self):
        reply = QtWidgets.QMessageBox.question(
            self, "保存图像", "处理完毕，是否保存结果图到本地？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.save_image()

    # ---------------- 色彩变换 ----------------
    def action_rgb2gray(self):
        if self.img is None:
            return
        self.push_undo()
        self.img = ImageProcessor.rgb_to_gray(self.img)
        self.show_image(self.img)
        self.offer_save_image()

    def action_gray2rgb(self):
        if self.img is None:
            return
        self.push_undo()
        self.img = ImageProcessor.gray_to_rgb(self.img)
        self.show_image(self.img)
        self.offer_save_image()

    # ---------------- 空域滤波 ----------------
    def action_mean_filter(self):
        if self.img is None:
            return
        self.push_undo()
        paramConfig = [
            {"name": "Kernel Size", "type": "int", "min": 3, "max": 31, "default": 3, "step": 2, "odd": True}
        ]
        explanation = (
            "【均值滤波原理】\n\n"
            "均值滤波是一种简单的空域平滑滤波方法。\n"
            "其原理是用局部窗口内所有像素的平均值替换中心像素，从而达到平滑图像、减少噪声的效果。\n"
            "但过大的滤波核会导致图像细节丢失。"
        )
        dlg = ParameterDialog(self, "均值滤波", self.img, ImageProcessor.mean_filter, paramConfig, explanation)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.img = dlg.getResultImage()
            self.show_image(self.img)
            self.offer_save_image()

    def action_gauss_filter(self):
        if self.img is None:
            return
        self.push_undo()
        paramConfig = [
            {"name": "Kernel Size", "type": "int", "min": 3, "max": 31, "default": 3, "step": 2, "odd": True},
            {"name": "Sigma", "type": "float", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1}
        ]
        explanation = (
            "【高斯滤波原理】\n\n"
            "高斯滤波利用高斯函数构造滤波核，\n"
            "核中心权值最大，向周围逐渐衰减，\n"
            "从而在平滑图像的同时较好地保留边缘信息。"
        )
        dlg = ParameterDialog(self, "高斯滤波", self.img, ImageProcessor.gaussian_filter, paramConfig, explanation)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.img = dlg.getResultImage()
            self.show_image(self.img)
            self.offer_save_image()

    def action_hist_eq(self):
        if self.img is None:
            return
        self.push_undo()
        # 直方图均衡化无需要调节的参数，因此参数列表置空
        paramConfig = []
        explanation = (
            "【直方图均衡化原理】\n\n"
            "直方图均衡化通过重新分布图像像素灰度值，\n"
            "增强图像对比度，使整体亮度分布更加均匀，\n"
            "从而改善图像细节和局部对比度。"
        )
        dlg = ParameterDialog(self, "直方图均衡化", self.img, ImageProcessor.hist_equalize, paramConfig, explanation)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.img = dlg.getResultImage()
            self.show_image(self.img)
            self.offer_save_image()

    # ---------------- 频域滤波 ----------------
    def action_butterworth_lowpass(self):
        if self.img is None:
            return
        self.push_undo()
        paramConfig = [
            {"name": "D0", "type": "int", "min": 1, "max": 200, "default": 30, "step": 1},
            {"name": "Order n", "type": "int", "min": 1, "max": 10, "default": 2, "step": 1}
        ]
        explanation = (
            "【巴特沃斯低通滤波原理】\n\n"
            "巴特沃斯低通滤波器在频域内平滑地衰减高频信号，\n"
            "参数D0决定保留低频成分的截止频率，\n"
            "阶数n影响滤波器的陡峭程度。"
        )
        dlg = ParameterDialog(self, "巴特沃斯低通滤波", self.img, ImageProcessor.butterworth_lowpass_filter,
                              paramConfig, explanation)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.img = dlg.getResultImage()
            self.show_image(self.img)
            self.offer_save_image()

    def action_gaussian_highpass(self):
        if self.img is None:
            return
        self.push_undo()
        paramConfig = [
            {"name": "D0", "type": "int", "min": 1, "max": 200, "default": 30, "step": 1}
        ]
        explanation = (
            "【高斯高通滤波原理】\n\n"
            "高斯高通滤波通过对高斯低通滤波器取补实现，\n"
            "能较好地突出图像中的细节和边缘信息，\n"
            "参数D0决定截止频率。"
        )
        dlg = ParameterDialog(self, "高斯高通滤波", self.img, ImageProcessor.gaussian_highpass_filter,
                              paramConfig, explanation)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.img = dlg.getResultImage()
            self.show_image(self.img)
            self.offer_save_image()

    # ---------------- 直方图显示 ----------------
    def action_compute_histogram(self):
        if self.img is None:
            return
        dlg = HistogramDialog(self, self.img, "RGB直方图")
        dlg.exec_()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
