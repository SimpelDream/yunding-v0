# API 文档

## 核心模块

### 状态模型

#### `Unit`

单位类，表示棋盘上的一个单位。

```python
class Unit(BaseModel):
    name: str
    star: Literal[1, 2, 3]
    items: List[str]
    position: Tuple[int, int]
    hp_pct: float
    shield_pct: float
    status_tags: List[str]
```

属性：
- `name`: 单位名称
- `star`: 星级(1-3)
- `items`: 装备列表
- `position`: 位置坐标(行0-7, 列0-3)
- `hp_pct`: 生命值百分比(0-1)
- `shield_pct`: 护盾百分比(0-1)
- `status_tags`: 状态标签列表

#### `BoardState`

棋盘状态类，表示当前游戏状态。

```python
class BoardState(BaseModel):
    side: Literal["self", "enemy"]
    stage: str
    phase_timer: Optional[int]
    rank: int
    hp: int
    gold: int
    level: int
    xp_progress: Tuple[int, int]
    shop_odds: Dict[str, float]
    shop_cards: List[str]
    traits: Dict[str, int]
    inactive_traits: Dict[str, int]
    units: List[Unit]
    bench_units: List[Unit]
    augments: List[str]
    hex_map: Dict[Tuple[int, int], str]
    combo_meter: Optional[int]
    enemy_hp_vec: List[int]
    fps: Optional[float]
    ping_ms: Optional[float]
    timestamp: float
```

属性：
- `side`: 阵营("self"或"enemy")
- `stage`: 游戏阶段(如"4-2")
- `phase_timer`: 阶段计时器
- `rank`: 当前排名
- `hp`: 生命值
- `gold`: 金币
- `level`: 等级
- `xp_progress`: 经验进度(当前/总需求)
- `shop_odds`: 商店概率
- `shop_cards`: 商店卡牌
- `traits`: 激活的特质
- `inactive_traits`: 未激活的特质
- `units`: 棋盘单位
- `bench_units`: 备战区单位
- `augments`: 海克斯强化
- `hex_map`: 六边形地图
- `combo_meter`: 连击计量器
- `enemy_hp_vec`: 敌人生命值向量
- `fps`: 帧率
- `ping_ms`: 延迟
- `timestamp`: 时间戳

### 检测器

#### `YOLONASDetector`

YOLO-NAS-S检测器类。

```python
class YOLONASDetector:
    def __init__(
        self,
        model_path: Union[str, Path],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """初始化检测器。"""
        
    def detect(self, image: np.ndarray) -> List[Dict]:
        """执行目标检测。"""
        
    def export_onnx(self, save_path: Union[str, Path]) -> None:
        """导出ONNX模型。"""
        
    @staticmethod
    def train(
        data_yaml: Union[str, Path],
        epochs: int = 60,
        batch_size: int = 16,
        imgsz: int = 1280,
        save_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """训练YOLO-NAS-S模型。"""
```

### 预测器

#### `LGBMPredictor`

LightGBM预测器类。

```python
class LGBMPredictor:
    def __init__(
        self,
        win_model_path: Optional[Union[str, Path]] = None,
        dmg_model_path: Optional[Union[str, Path]] = None
    ) -> None:
        """初始化预测器。"""
        
    def build_feature(self, state: BoardState) -> np.ndarray:
        """构建特征向量。"""
        
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """预测胜率和伤害。"""
        
    @staticmethod
    def train(
        X: np.ndarray,
        y_win: np.ndarray,
        y_dmg: np.ndarray,
        save_dir: Union[str, Path],
        **kwargs
    ) -> None:
        """训练模型。"""
```

## 插件系统

### 钩子函数

#### `on_state_update`

当游戏状态更新时触发。

```python
@hookimpl
def on_state_update(state: BoardState, ui: UI) -> None:
    """处理状态更新。"""
```

参数：
- `state`: 当前游戏状态
- `ui`: UI实例

#### `on_game_end`

当游戏结束时触发。

```python
@hookimpl
def on_game_end(history: List[BoardState]) -> None:
    """处理游戏结束。"""
```

参数：
- `history`: 游戏历史状态列表

## 工具函数

### 图像处理

```python
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """预处理图像。"""
    
def postprocess_detections(detections: List[Dict]) -> List[Dict]:
    """后处理检测结果。"""
```

### 特征工程

```python
def extract_unit_features(units: List[Unit]) -> List[float]:
    """提取单位特征。"""
    
def extract_hex_features(hex_map: Dict[Tuple[int, int], str]) -> List[float]:
    """提取海克斯特征。"""
```

### 数据管理

```python
def load_model(model_path: Union[str, Path]) -> Any:
    """加载模型。"""
    
def save_model(model: Any, save_path: Union[str, Path]) -> None:
    """保存模型。"""
``` 