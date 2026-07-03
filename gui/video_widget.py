"""
GUI - 视频画面组件 (Video Widget)
职责：显示摄像头实时画面，骨架关键点与连线叠加渲染
"""

import cv2
import numpy as np
from typing import Optional
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont
import logging

logger = logging.getLogger(__name__)


class VideoWidget(QWidget):
    """视频显示组件，支持骨架叠加渲染"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #1a1a1a;")

        self._image: Optional[np.ndarray] = None
        self._skeleton_image: Optional[np.ndarray] = None
        self._fps = 0.0
        self._frame_width = 0
        self._frame_height = 0

    def update_frame(self, image: np.ndarray, skeleton_image: np.ndarray = None,
                     fps: float = 0.0):
        """
        更新显示帧

        Args:
            image: 原始 BGR 图像
            skeleton_image: 带骨架叠加的图像（可选）
            fps: 当前帧率
        """
        self._image = image
        self._fps = fps
        if skeleton_image is not None:
            self._skeleton_image = skeleton_image
        else:
            self._skeleton_image = image.copy()

        if image is not None:
            self._frame_width = image.shape[1]
            self._frame_height = image.shape[0]

        self.update()  # 触发 paintEvent

    def paintEvent(self, event):
        """绘制事件"""
        if self._skeleton_image is None:
            # 无画面时显示文字
            painter = QPainter(self)
            painter.setPen(QColor(128, 128, 128))
            painter.setFont(QFont("Microsoft YaHei", 16))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待摄像头连接...")
            painter.end()
            return

        # 将 OpenCV BGR 转为 QImage RGB
        rgb = cv2.cvtColor(self._skeleton_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 缩放适应窗口（保持比例）
        pix = QPixmap.fromImage(q_img)
        scaled = pix.scaled(self.size(), Qt.KeepAspectRatio,
                            Qt.SmoothTransformation)

        # 居中绘制
        painter = QPainter(self)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)

        # 绘制 FPS 信息（左上角）
        if self._fps > 0:
            painter.setPen(QColor(0, 255, 0))
            painter.setFont(QFont("Consolas", 12))
            painter.drawText(x + 8, y + 22, f"FPS: {self._fps:.1f}")

        painter.end()
