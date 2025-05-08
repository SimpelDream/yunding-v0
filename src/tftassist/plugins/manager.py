"""插件管理器模块。

此模块实现了插件的加载和管理功能。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import pluggy

logger = logging.getLogger(__name__)

class PluginManager:
    """插件管理器类。"""

    def __init__(self, plugin_dir: str):
        """初始化插件管理器。

        Args:
            plugin_dir: 插件目录路径
        """
        self.plugin_dir = Path(plugin_dir)
        self.pm = pluggy.PluginManager("tftassist")
        logger.info(f"初始化插件管理器: {self.plugin_dir}")

    def load_plugins(self) -> None:
        """加载所有插件。"""
        # TODO: 实现插件加载逻辑
        pass

    def get_plugin(self, name: str) -> Any:
        """获取指定名称的插件。

        Args:
            name: 插件名称

        Returns:
            插件对象
        """
        return self.pm.get_plugin(name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """列出所有已加载的插件。

        Returns:
            插件信息列表
        """
        plugins = []
        for name, plugin in self.pm.list_name_plugin():
            plugins.append({
                "name": name,
                "version": getattr(plugin, "__version__", "unknown"),
                "description": getattr(plugin, "__doc__", "")
            })
        return plugins 