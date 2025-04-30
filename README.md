# TFT 云顶之弈辅助工具

[![CI](https://github.com/yourusername/tft-assist/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/tft-assist/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/yourusername/tft-assist/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/tft-assist)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TFT 云顶之弈辅助工具，使用 YOLO-NAS 和 PaddleOCR 解析游戏画面，LightGBM 预测胜率和伤害。

## 功能特点

- 🎯 实时屏幕捕获
- 🔍 YOLO-NAS 目标检测
- 📝 PaddleOCR 文本识别
- 📊 LightGBM 胜率预测
- 💥 伤害预测
- 🖥️ 游戏内悬浮窗
- 🔌 插件系统支持

## 安装

### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/tft-assist.git
cd tft-assist

# 安装依赖
poetry install

# 安装项目
poetry install --no-dev
```

### 从 PyPI 安装

```bash
pip install tft-assist
```

## 使用方法

### 基本使用

```bash
# 运行程序
python -m tftassist

# 演示模式
python -m tftassist --demo
```

### 插件开发

1. 在 `plugins` 目录下创建 Python 文件
2. 实现 `on_state_update` 和 `on_game_end` 钩子
3. 重启程序加载插件

示例插件：

```python
from tftassist.plugins.hooks import PluginHookSpec

class MyPlugin:
    def on_state_update(self, state, ui):
        # 处理状态更新
        pass

    def on_game_end(self, history):
        # 处理游戏结束
        pass
```

## 开发

### 环境设置

```bash
# 安装开发依赖
poetry install

# 安装预提交钩子
pre-commit install
```

### 训练模型

```bash
# 训练检测模型
python scripts/train_det.py

# 训练预测模型
python scripts/train_lgbm.py
```

### 测试

```bash
# 运行测试
pytest

# 检查覆盖率
pytest --cov=tftassist --cov-report=term-missing
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License 