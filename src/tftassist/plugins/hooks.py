"""插件系统钩子定义。

此模块定义了插件系统的主要钩子，用于扩展程序功能。
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar

import pluggy

from tftassist.core.state import BoardState
from tftassist.ui.overlay import OverlayWindow

logger = logging.getLogger(__name__)

# 定义钩子管理器
hookspec = pluggy.HookspecMarker("tftassist")
hookimpl = pluggy.HookimplMarker("tftassist")

# 定义类型变量
T = TypeVar("T")

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

class TFTAssistSpec(Protocol):
    """TFT助手钩子规范。"""
    
    @hookspec
    def on_game_start(self) -> None:
        """游戏开始时调用。"""
        ...
        
    @hookspec
    def on_game_end(self) -> None:
        """游戏结束时调用。"""
        ...
        
    @hookspec
    def on_state_update(self, state: Any) -> None:
        """游戏状态更新时调用。
        
        Args:
            state: 游戏状态对象
        """
        ...
        
    @hookspec
    def on_prediction(self, prediction: Any) -> None:
        """预测结果更新时调用。
        
        Args:
            prediction: 预测结果对象
        """
        ...

class HookManager:
    """钩子管理器。"""
    
    def __init__(self) -> None:
        """初始化钩子管理器。"""
        self.manager = pluggy.PluginManager("tftassist")
        self.manager.add_hookspecs(TFTAssistSpec)
        
    def register_plugin(self, plugin: Any) -> None:
        """注册插件。
        
        Args:
            plugin: 插件对象
        """
        self.manager.register(plugin)
        logger.info(f"已注册插件: {plugin.__class__.__name__}")
        
    def unregister_plugin(self, plugin: Any) -> None:
        """注销插件。
        
        Args:
            plugin: 插件对象
        """
        self.manager.unregister(plugin)
        logger.info(f"已注销插件: {plugin.__class__.__name__}")
        
    def get_plugins(self) -> List[Any]:
        """获取所有已注册的插件。
        
        Returns:
            插件列表
        """
        return list(self.manager.get_plugins())
        
    def call_hook(self, hook_name: str, **kwargs: Any) -> List[Any]:
        """调用钩子。
        
        Args:
            hook_name: 钩子名称
            **kwargs: 钩子参数
            
        Returns:
            钩子返回值列表
        """
        return self.manager.hook.__getattr__(hook_name)(**kwargs) 