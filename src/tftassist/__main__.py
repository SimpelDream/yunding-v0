"""TFT助手主程序。"""

import logging
import sys
from typing import Optional

from PySide6.QtWidgets import QApplication

from .ui.overlay import OverlayWindow

logger = logging.getLogger(__name__)

def main() -> None:
    """主程序入口。"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 创建应用
    app = QApplication(sys.argv)
    
    # 创建窗口
    window = OverlayWindow()
    window.show()
    
    # 运行应用
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 