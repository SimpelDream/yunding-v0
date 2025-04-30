"""屏幕截图模块。

此模块负责捕获游戏画面，支持多种截图方式。
"""

import asyncio
import logging
from typing import Optional

import mss
import numpy as np
from mss.base import ScreenShot

logger = logging.getLogger(__name__)


class ScreenCapturer:
    """屏幕截图器。

    使用 mss 库进行屏幕截图，支持异步操作。
    """

    def __init__(self, monitor: int = 1) -> None:
        """初始化截图器。

        Args:
            monitor: 显示器编号，默认为主显示器
        """
        self.monitor = monitor
        self.sct = mss.mss()
        self._last_frame: Optional[np.ndarray] = None

    async def capture(self) -> np.ndarray:
        """异步捕获屏幕画面。

        Returns:
            当前屏幕画面的 numpy 数组
        """
        # 使用线程池执行阻塞的截图操作
        loop = asyncio.get_event_loop()
        frame = await loop.run_in_executor(None, self._capture_sync)
        self._last_frame = frame
        return frame

    def _capture_sync(self) -> np.ndarray:
        """同步捕获屏幕画面。

        Returns:
            当前屏幕画面的 numpy 数组
        """
        try:
            screenshot: ScreenShot = self.sct.grab(self.sct.monitors[self.monitor])
            # 将 BGRA 转换为 BGR
            frame = np.array(screenshot)[:, :, :3]
            return frame
        except Exception as e:
            logger.error(f"截图失败: {e}")
            if self._last_frame is not None:
                return self._last_frame
            raise

    def close(self) -> None:
        """关闭截图器。"""
        self.sct.close() 