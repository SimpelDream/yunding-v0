"""插件系统钩子定义。

此模块定义了插件系统的主要钩子，用于扩展程序功能。
"""

from typing import Protocol

from tftassist.core.state import BoardState
from tftassist.ui.overlay import OverlayWindow


class PluginHookSpec(Protocol):
    """插件钩子规范。

    定义了插件系统支持的所有钩子方法。
    """

    def on_state_update(self, state: BoardState, ui: OverlayWindow) -> None:
        """当游戏状态更新时调用。

        Args:
            state: 当前游戏状态
            ui: 悬浮窗UI实例
        """
        ...

    def on_game_end(self, history: list[BoardState]) -> None:
        """当游戏结束时调用。

        Args:
            history: 游戏历史状态列表
        """
        ... 