"""游戏状态模块。

此模块定义了游戏状态的数据结构，包括单位、棋盘状态等核心数据模型。
使用pydantic进行数据验证和序列化。
"""

from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field
from dataclasses import dataclass, field

@dataclass
class Unit:
    """游戏单位。
    
    Attributes:
        name: 单位名称
        star: 星级
        items: 装备列表
        position: 位置坐标
        hp_pct: 生命值百分比
        shield_pct: 护盾值百分比
        status_tags: 状态标签列表
    """
    
    name: str
    star: int
    items: List[str]
    position: Tuple[int, int]
    hp_pct: float
    shield_pct: float
    status_tags: List[str] = field(default_factory=list)

@dataclass
class BoardState:
    """游戏状态。
    
    Attributes:
        side: 阵营
        stage: 阶段
        phase_timer: 阶段计时器
        rank: 排名
        hp: 生命值
        gold: 金币
        level: 等级
        xp_progress: 经验值进度
        shop_odds: 商店概率
        shop_cards: 商店卡牌
        traits: 激活的特质
        inactive_traits: 未激活的特质
        units: 场上单位
        bench_units: 备战区单位
        augments: 强化符文
        hex_map: 六边形地图
        combo_meter: 连击计量器
        enemy_hp_vec: 敌人生命值向量
        fps: 帧率
        ping_ms: 延迟
        timestamp: 时间戳
    """
    
    side: str
    stage: str
    rank: int
    hp: int
    gold: int
    level: int
    xp_progress: Tuple[int, int]
    timestamp: float
    
    phase_timer: float = 0.0
    shop_odds: Dict[str, float] = field(default_factory=dict)
    shop_cards: List[str] = field(default_factory=list)
    traits: Dict[str, int] = field(default_factory=dict)
    inactive_traits: Dict[str, int] = field(default_factory=dict)
    units: List[Unit] = field(default_factory=list)
    bench_units: List[Unit] = field(default_factory=list)
    augments: List[str] = field(default_factory=list)
    hex_map: Dict[Tuple[int, int], str] = field(default_factory=dict)
    combo_meter: int = 0
    enemy_hp_vec: List[int] = field(default_factory=list)
    fps: int = 0
    ping_ms: int = 0

# 特质词汇表
TRAIT_VOCAB: Dict[str, str] = {
    "Mage": "法师",
    "Assassin": "刺客",
    "Tank": "坦克",
    "Support": "辅助",
    "Marksman": "射手",
    "Fighter": "战士",
    "Mage": "法师",
    "Assassin": "刺客",
    "Tank": "坦克",
    "Support": "辅助",
    "Marksman": "射手",
    "Fighter": "战士"
} 