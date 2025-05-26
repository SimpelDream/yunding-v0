# TFT Assistant

云顶之弈助手，基于计算机视觉和机器学习的游戏辅助工具。

## 快速开始

### Windows

```powershell
.\scripts\setup_env.ps1
.\scripts\dvc_flow.ps1
```

### Linux/macOS

```bash
bash scripts/setup_env.sh
bash scripts/dvc_flow.sh
```

## 功能特性

- 实时游戏状态识别
- 阵容推荐
- 装备推荐
- 海克斯强化建议
- 对局数据分析

## 开发环境

- Python 3.10+
- Poetry
- CUDA 11.8+ (可选)

## 项目结构

```
tftassist/
├── src/
│   └── tftassist/
│       ├── capture/    # 屏幕捕获
│       ├── core/       # 核心功能
│       ├── detector/   # 目标检测
│       ├── ocr/        # 文字识别
│       ├── predictor/  # 预测模型
│       └── ui/         # 用户界面
├── scripts/            # 工具脚本
├── tests/             # 测试用例
└── models/            # 模型文件
```

## 开发指南

1. 克隆仓库
2. 安装依赖
3. 运行测试
4. 开始开发

## 贡献指南

1. Fork 仓库
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 许可证

MIT License 