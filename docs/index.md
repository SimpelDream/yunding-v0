# 云顶之弈助手 (TFT Assist)

基于YOLO-NAS和LightGBM的云顶之弈游戏状态分析和胜率预测工具。

## 功能特点

- 🎯 实时游戏状态检测
  - 使用YOLO-NAS-S检测棋盘单位、装备、状态等
  - 使用PaddleOCR-lite识别文本信息
  - 支持海克斯、传送门等特殊元素识别

- 📊 胜率预测
  - 基于LightGBM的胜率和伤害预测
  - 考虑单位、装备、特质等多维度特征
  - 实时更新预测结果

- 🖥️ 悬浮窗显示
  - 使用PySide6构建现代化UI
  - 支持自定义主题和布局
  - 低资源占用，不影响游戏体验

- 🔌 插件系统
  - 基于Pluggy的插件架构
  - 支持自定义数据分析和可视化
  - 丰富的插件生态

## 安装

```bash
# 使用pip安装
pip install tft-assist

# 使用Poetry安装（推荐）
poetry add tft-assist

# 安装DXCam支持（可选）
poetry add tft-assist[dxcam]
```

## 快速开始

1. 启动游戏
2. 运行助手
```bash
python -m tftassist
```
3. 享受实时分析和预测

## 开发指南

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/yourusername/tft-assist.git
cd tft-assist

# 安装依赖
poetry install

# 安装开发工具
poetry install --with dev
```

### 训练模型

```bash
# 训练检测器
python scripts/train_det.py --data datasets/data.yaml

# 训练预测器
python scripts/train_lgbm.py --data datasets/train.csv
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行带覆盖率报告的测试
pytest --cov=src/tftassist
```

## 插件开发

1. 创建插件目录
```bash
mkdir -p ~/.tftassist/plugins
```

2. 创建插件文件
```python
# my_plugin.py
import pluggy

hookimpl = pluggy.HookimplMarker("tftassist")

@hookimpl
def on_state_update(state, ui):
    # 处理状态更新
    pass

@hookimpl
def on_game_end(history):
    # 处理游戏结束
    pass
```

3. 安装插件
```bash
cp my_plugin.py ~/.tftassist/plugins/
```

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件 