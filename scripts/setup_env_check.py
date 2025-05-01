"""环境检查脚本。

此脚本检查项目依赖的版本信息。
"""

import sys
import subprocess
from typing import Dict, List

def get_version(cmd: str) -> str:
    """获取命令的版本信息。

    Args:
        cmd: 要检查的命令

    Returns:
        版本信息字符串
    """
    try:
        # 使用 shell=True 以支持 Windows 命令
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "未安装"

def main() -> None:
    """主函数，打印所有依赖的版本信息。"""
    # 要检查的命令列表
    commands: Dict[str, str] = {
        "Python": "python --version",
        "Poetry": "poetry --version",
        "Ruff": "ruff --version",
        "Mypy": "mypy --version",
        "Pytest": "pytest --version"
    }
    
    # 打印版本信息
    print("依赖版本检查:")
    print("-" * 50)
    for name, cmd in commands.items():
        version = get_version(cmd)
        print(f"{name}: {version}")
    print("-" * 50)

if __name__ == "__main__":
    main() 