"""游戏界面叠加层。"""

import logging
from typing import Optional

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPainter, QPen, QCloseEvent, QPaintEvent
from PySide6.QtWidgets import QMainWindow, QWidget

from ..core.state import BoardState

logger = logging.getLogger(__name__)

class OverlayWindow(QMainWindow):
    """游戏界面叠加窗口。
    
    此窗口用于显示游戏状态和预测结果, 支持透明背景和置顶显示。
    """
    
    def __init__(self) -> None:
        """初始化叠加窗口。"""
        super().__init__()
        
        # 设置窗口属性
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 创建中央部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 初始化状态
        self.state = BoardState(
            side="self",
            stage="1-1",
            rank=1,
            hp=100,
            gold=0,
            level=1,
            xp_progress=(0, 2),
            timestamp=0.0
        )
        
        # 创建定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(100)  # 10 FPS
        
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
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景
        painter.fillRect(self.rect(), Qt.transparent)
        
        # 绘制边框
        pen = QPen(Qt.red, 2)
        painter.setPen(pen)
        painter.drawRect(self.rect().adjusted(1, 1, -1, -1))
        
        # 绘制状态信息
        painter.setPen(Qt.white)
        painter.drawText(10, 20, f"HP: {self.state.hp}")
        painter.drawText(10, 40, f"Gold: {self.state.gold}")
        painter.drawText(10, 60, f"Level: {self.state.level}")
        painter.drawText(10, 80, f"Stage: {self.state.stage}")
        
    def update_state(self, state: BoardState) -> None:
        """更新游戏状态。
        
        Args:
            state: 新的游戏状态
        """
        self.state = state
        self.update() 