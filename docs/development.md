# 开发指南

## 环境设置

### 系统要求

- Python 3.8+
- CUDA 11.0+ (用于GPU加速)
- Git

### 开发环境

1. 克隆仓库
```bash
git clone https://github.com/yourusername/tft-assist.git
cd tft-assist
```

2. 安装Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. 安装依赖
```bash
poetry install
poetry install --with dev
```

4. 安装预提交钩子
```bash
poetry run pre-commit install
```

## 代码风格

本项目使用以下工具确保代码质量：

- Black: 代码格式化
- isort: import排序
- ruff: 代码检查
- mypy: 类型检查

### 运行检查

```bash
# 格式化代码
poetry run black src tests

# 排序imports
poetry run isort src tests

# 运行代码检查
poetry run ruff check src tests

# 运行类型检查
poetry run mypy src tests
```

## 测试

### 运行测试

```bash
# 运行所有测试
poetry run pytest

# 运行带覆盖率报告的测试
poetry run pytest --cov=src/tftassist

# 运行特定测试
poetry run pytest tests/test_state.py
```

### 添加测试

1. 在`tests`目录下创建测试文件
2. 使用`pytest`装饰器和断言
3. 确保测试覆盖率达到80%以上

示例：
```python
def test_unit_creation():
    unit = Unit(
        name="test_unit",
        star=2,
        items=["item1"],
        position=(3, 2),
        hp_pct=0.8,
        shield_pct=0.3,
        status_tags=[]
    )
    assert unit.name == "test_unit"
    assert unit.star == 2
```

## 模型训练

### 数据准备

1. 创建数据目录
```bash
mkdir -p datasets/raw
```

2. 添加数据文件
```bash
# 检测器数据
datasets/raw/
  ├─ images/
  │   ├─ train/
  │   └─ val/
  └─ labels/
      ├─ train/
      └─ val/

# 预测器数据
datasets/raw/train.csv
```

3. 配置DVC
```bash
dvc add datasets/raw
```

### 训练检测器

```bash
# 训练YOLO-NAS-S模型
poetry run python scripts/train_det.py \
    --data datasets/raw/data.yaml \
    --epochs 60 \
    --batch-size 16 \
    --img-size 1280
```

### 训练预测器

```bash
# 训练LightGBM模型
poetry run python scripts/train_lgbm.py \
    --data datasets/raw/train.csv \
    --num-leaves 96 \
    --n-estimators 1200 \
    --learning-rate 0.03
```

## 插件开发

### 创建插件

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

### 插件API

#### 钩子函数

- `on_state_update(state, ui)`: 状态更新时触发
- `on_game_end(history)`: 游戏结束时触发

#### 工具函数

- `preprocess_image(image)`: 图像预处理
- `postprocess_detections(detections)`: 检测结果后处理
- `extract_unit_features(units)`: 单位特征提取
- `extract_hex_features(hex_map)`: 海克斯特征提取

## 发布流程

1. 更新版本号
```bash
poetry version patch  # 或 minor/major
```

2. 运行测试
```bash
poetry run pytest
```

3. 构建包
```bash
poetry build
```

4. 发布到PyPI
```bash
poetry publish
```

5. 创建GitHub Release
```bash
git tag v0.1.0
git push origin v0.1.0
```

## 贡献指南

1. Fork 项目
2. 创建特性分支
```bash
git checkout -b feature/my-feature
```

3. 提交更改
```bash
git commit -m "Add my feature"
```

4. 推送到分支
```bash
git push origin feature/my-feature
```

5. 创建 Pull Request

### Pull Request 检查清单

- [ ] 代码符合项目风格指南
- [ ] 添加了必要的测试
- [ ] 更新了文档
- [ ] 所有测试通过
- [ ] 代码覆盖率达标 