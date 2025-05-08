"""游戏界面叠加层。"""

import logging
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QPainter, QPen, Qt, QCloseEvent, QPaintEvent
from PySide6.QtWidgets import QMainWindow, QWidget

from ..core.state import BoardState, Unit

logger = logging.getLogger(__name__)

class OverlayWindow(QMainWindow):
    """叠加窗口类。"""
    
    def __init__(self) -> None:
        """初始化叠加窗口。"""
        super().__init__()
        
        # 设置窗口属性
        self.setWindowTitle("TFT Assistant")
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # 创建中心部件
        self.central_widget = OverlayWidget()
        self.setCentralWidget(self.central_widget)
        
        # 设置定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_overlay)
        self.timer.start(1000 // 60)  # 60 FPS
        
    def update_overlay(self) -> None:
        """更新叠加层。"""
        self.central_widget.update()
        
    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭事件处理。"""
        self.timer.stop()
        event.accept()

class OverlayWidget(QWidget):
    """叠加部件类。"""
    
    def __init__(self) -> None:
        """初始化叠加部件。"""
        super().__init__()
        
        # 初始化游戏状态
        self.state = BoardState()
        
    def paintEvent(self, event: QPaintEvent) -> None:
        """绘制事件处理。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 设置画笔
        pen = QPen(Qt.GlobalColor.green)
        pen.setWidth(2)
        painter.setPen(pen)
        
        # 绘制叠加内容
        # TODO: 实现叠加内容的绘制
        
    def update_state(self, state: BoardState) -> None:
        """更新游戏状态。"""
        self.state = state
        self.update() 