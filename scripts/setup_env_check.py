"""环境检查脚本。

此脚本用于检查项目依赖的版本信息。
"""

import subprocess
from typing import Optional

def get_package_version(package: str) -> Optional[str]:
    """获取包版本。
    
    Args:
        package: 包名
        
    Returns:
        版本字符串, 如果包未安装则返回None
    """
    try:
        result = subprocess.run(
            ["pip", "show", package],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.splitlines():
            if line.startswith("Version: "):
                return line.split(": ")[1]
        return None
    except subprocess.CalledProcessError:
        return None

def main() -> None:
    """主函数, 打印所有依赖的版本信息。"""
    # 核心依赖
    core_deps = [
        "torch",
        "numpy",
        "pandas",
        "lightgbm",
        "ultralytics",
        "paddleocr",
        "pyside6",
        "pluggy"
    ]
    
    # 开发依赖
    dev_deps = [
        "pytest",
        "mypy",
        "ruff",
        "pre-commit"
    ]
    
    print("核心依赖:")
    for dep in core_deps:
        version = get_package_version(dep)
        if version:
            print(f"  {dep}: {version}")
        else:
            print(f"  {dep}: 未安装")
            
    print("\n开发依赖:")
    for dep in dev_deps:
        version = get_package_version(dep)
        if version:
            print(f"  {dep}: {version}")
        else:
            print(f"  {dep}: 未安装")

if __name__ == "__main__":
    main() 