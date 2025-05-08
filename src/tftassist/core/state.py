"""游戏状态模块。

此模块定义了游戏状态的数据结构。
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class Unit:
    """单位类。"""
    
    name: str
    traits: List[str]
    position: Tuple[int, int]
    size: Tuple[int, int]
    level: int = 1
    hp: float = 100.0
    attack: float = 10.0
    defense: float = 5.0
    attack_speed: float = 1.0
    range: int = 1
    mana: float = 0.0
    max_mana: float = 100.0
    crit_chance: float = 0.0
    dodge_chance: float = 0.0

class BoardState:
    """棋盘状态类。"""
    
    def __init__(self) -> None:
        """初始化棋盘状态。"""
        # 基本信息
        self.side: str = ""  # 玩家阵营
        self.stage: int = 1  # 当前阶段
        self.phase_timer: float = 0.0  # 阶段计时器
        self.rank: int = 0  # 当前排名
        self.hp: int = 100  # 生命值
        self.gold: int = 0  # 金币
        self.level: int = 1  # 等级
        self.xp_progress: float = 0.0  # 经验进度
        
        # 商店信息
        self.shop_odds: Dict[str, float] = {}  # 商店概率
        self.shop_cards: List[Unit] = []  # 商店卡牌
        
        # 特质信息
        self.traits: Dict[str, int] = {}  # 激活的特质
        self.inactive_traits: Dict[str, int] = {}  # 未激活的特质
        
        # 单位信息
        self.units: List[Unit] = []  # 棋盘单位
        self.bench_units: List[Unit] = []  # 备战区单位
        
        # 海克斯信息
        self.augments: List[str] = []  # 海克斯强化
        
        # 棋盘信息
        self.hex_map: np.ndarray = np.zeros((0, 0), dtype=np.uint8)  # 六边形地图
        self.combo_meter: float = 0.0  # 连击计量器
        
        # 敌人信息
        self.enemy_hp_vec: np.ndarray = np.zeros(8, dtype=np.float32)  # 敌人生命值向量
        
        # 性能信息
        self.fps: float = 0.0  # 帧率
        self.ping_ms: float = 0.0  # 延迟
        self.timestamp: float = 0.0  # 时间戳
        
        # 尺寸信息
        self.board_height: int = 0  # 棋盘高度
        self.board_width: int = 0  # 棋盘宽度
        
        # 状态信息
        self.hp_pct: float = 1.0  # 生命值百分比
        self.shield_pct: float = 0.0  # 护盾百分比

# 特质词汇表
TRAIT_VOCAB = {
    "assassin": "刺客",
    "brawler": "格斗家",
    "challenger": "挑战者",
    "enchanter": "秘术师",
    "gunner": "枪手",
    "mage": "法师",
    "marksman": "射手",
    "tank": "坦克",
    "vanguard": "先锋"
} 