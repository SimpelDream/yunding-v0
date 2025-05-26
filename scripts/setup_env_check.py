"""环境检查脚本。

此脚本用于检查项目依赖的版本和配置。
"""

import importlib
import sys
from typing import Dict, Optional

def get_package_version(package_name: str) -> Optional[str]:
    """获取包的版本。

    Args:
        package_name: 包名

    Returns:
        版本字符串，如果包未安装则返回 None
    """
    try:
        module = importlib.import_module(package_name)
        return getattr(module, "__version__", None)
    except ImportError:
        return None

def check_dependencies() -> Dict[str, Optional[str]]:
    """检查主要依赖的版本。

    Returns:
        依赖版本字典
    """
    dependencies = {
        "numpy": None,
        "opencv-python": None,
        "onnxruntime": None,
        "ultralytics": None,
        "paddlepaddle": None,
        "paddleocr": None,
        "lightgbm": None,
        "dxcam": None,
        "pyside6": None,
        "numba": None,
        "duckdb": None,
        "dvc": None
    }
    
    for package in dependencies:
        dependencies[package] = get_package_version(package)
        
    return dependencies

def main() -> None:
    """主函数。"""
    print("Python 版本:", sys.version)
    print("\n依赖版本:")
    
    deps = check_dependencies()
    for package, version in deps.items():
        status = f"✅ {version}" if version else "❌ 未安装"
        print(f"{package}: {status}")

if __name__ == "__main__":
    main() 