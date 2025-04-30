"""悬浮窗 UI 模块。

此模块实现了游戏内悬浮窗界面。
"""

import logging
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

from tftassist.core.state import BoardState

logger = logging.getLogger(__name__)


class OverlayWindow(QMainWindow):
    """游戏内悬浮窗。

    显示胜率、伤害等游戏信息。
    """

    def __init__(self) -> None:
        """初始化悬浮窗。"""
        super().__init__()
        
        # 设置窗口属性
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint
            | Qt.FramelessWindowHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 创建中央部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 初始化状态
        self.win_rate: Optional[float] = None
        self.damage: Optional[float] = None
        self.state: Optional[BoardState] = None
        
        # 设置定时器用于更新
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(100)  # 10 FPS
        
        logger.info("初始化悬浮窗")

    def update_state(self, win_rate: float, damage: float, state: BoardState) -> None:
        """更新显示状态。

        Args:
            win_rate: 胜率
            damage: 伤害
            state: 游戏状态
        """
        self.win_rate = win_rate
        self.damage = damage
        self.state = state
        self.update()

    def paintEvent(self, event) -> None:
        """绘制事件处理。

        Args:
            event: 绘制事件
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 设置半透明背景
        painter.fillRect(self.rect(), QColor(0, 0, 0, 128))
        
        # 设置文本颜色
        painter.setPen(QPen(Qt.white))
        
        # 绘制胜率和伤害
        if self.win_rate is not None and self.damage is not None:
            text = f"胜率: {self.win_rate:.1%}\n伤害: {self.damage:.1f}"
            painter.drawText(self.rect(), Qt.AlignCenter, text)
        
        # 绘制游戏状态
        if self.state is not None:
            # TODO: 实现更多状态信息的绘制
            pass

    def mousePressEvent(self, event) -> None:
        """鼠标按下事件处理。

        Args:
            event: 鼠标事件
        """
        # 允许拖动窗口
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        """鼠标移动事件处理。

        Args:
            event: 鼠标事件
        """
        # 处理窗口拖动
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def closeEvent(self, event) -> None:
        """关闭事件处理。

        Args:
            event: 关闭事件
        """
        self.update_timer.stop()
        super().closeEvent(event) 