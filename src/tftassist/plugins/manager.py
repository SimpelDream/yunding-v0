"""插件管理器。

此模块实现了插件系统的核心功能，负责插件的加载和管理。
"""

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, List, Type

import pluggy

from tftassist.plugins.hooks import PluginHookSpec

logger = logging.getLogger(__name__)


class PluginManager:
    """插件管理器类。

    负责插件的加载、注册和管理。
    """

    def __init__(self) -> None:
        """初始化插件管理器。"""
        self.pm = pluggy.PluginManager("tftassist")
        self.pm.add_hookspecs(PluginHookSpec)
        self._plugins: Dict[str, Any] = {}

    def load_plugin(self, plugin_path: Path) -> None:
        """加载单个插件。

        Args:
            plugin_path: 插件文件路径
        """
        try:
            spec = importlib.util.spec_from_file_location(
                plugin_path.stem, str(plugin_path)
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"无法加载插件: {plugin_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.pm.register(module)
            self._plugins[plugin_path.stem] = module
            logger.info(f"成功加载插件: {plugin_path.stem}")
        except Exception as e:
            logger.error(f"加载插件失败 {plugin_path}: {e}")

    def load_plugins(self, plugin_dir: Path) -> None:
        """加载指定目录下的所有插件。

        Args:
            plugin_dir: 插件目录路径
        """
        if not plugin_dir.exists():
            logger.warning(f"插件目录不存在: {plugin_dir}")
            return

        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            self.load_plugin(plugin_file)

    def get_plugin(self, name: str) -> Any:
        """获取指定名称的插件实例。

        Args:
            name: 插件名称

        Returns:
            插件实例
        """
        return self._plugins.get(name)

    def get_all_plugins(self) -> Dict[str, Any]:
        """获取所有已加载的插件。

        Returns:
            插件名称到插件实例的映射
        """
        return self._plugins.copy()

    def call_hook(self, hook_name: str, *args: Any, **kwargs: Any) -> List[Any]:
        """调用指定名称的钩子。

        Args:
            hook_name: 钩子名称
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            所有插件钩子的返回值列表
        """
        return self.pm.hook.__getattr__(hook_name)(*args, **kwargs) 