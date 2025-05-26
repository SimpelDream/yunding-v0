"""游戏界面叠加层。"""

import logging
from typing import Optional

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPainter, QPen, QCloseEvent, QPaintEvent
from PySide6.QtWidgets import QMainWindow, QWidget

from ..core.state import BoardState, Unit

logger = logging.getLogger(__name__)

class OverlayWindow(QMainWindow):
    """游戏界面叠加窗口。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """初始化叠加窗口。

        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # 创建定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_overlay)
        self.timer.start(16)  # ~60 FPS
        
        # 状态
        self.state = BoardState()
        
    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭事件处理。

        Args:
            event: 关闭事件
        """
        self.timer.stop()
        super().closeEvent(event)
        
    def paintEvent(self, event: QPaintEvent) -> None:
        """绘制事件处理。

        Args:
            event: 绘制事件
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制边框
        pen = QPen(Qt.GlobalColor.red, 2)
        painter.setPen(pen)
        painter.drawRect(self.rect())
        
        # 绘制单位
        for unit in self.state.units:
            x, y = unit.position
            w, h = unit.size
            painter.drawRect(x, y, w, h)
            
        # 绘制备战区
        for unit in self.state.bench_units:
            x, y = unit.position
            w, h = unit.size
            painter.drawRect(x, y, w, h)

    def update_overlay(self) -> None:
        """更新叠加层。"""
        self.update() 