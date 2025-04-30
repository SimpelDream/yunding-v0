"""TFT 游戏状态数据模型定义。

此模块定义了游戏状态的核心数据模型，包括棋子单位和棋盘状态。
"""

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel


class Unit(BaseModel):
    """棋子单位数据模型。

    Attributes:
        name: 棋子名称
        star: 星级 (1-3)
        items: 装备列表
        position: 位置坐标 (行0-7, 列0-3)
        hp_pct: 生命值百分比 (0-1)
        shield_pct: 护盾值百分比 (0-1)
        status_tags: 状态标签列表
    """

    name: str
    star: Literal[1, 2, 3]
    items: list[str]
    position: tuple[int, int]  # 行0-7, 列0-3
    hp_pct: float  # 0-1
    shield_pct: float  # 0-1
    status_tags: list[str]


class BoardState(BaseModel):
    """棋盘状态数据模型。

    Attributes:
        side: 己方/敌方
        stage: 游戏阶段 (如 "4-2")
        phase_timer: 阶段剩余时间
        rank: 当前排名
        hp: 生命值
        gold: 金币
        level: 等级
        xp_progress: 经验值进度 (当前/总)
        shop_odds: 商店概率
        shop_cards: 商店卡牌
        traits: 已激活羁绊
        inactive_traits: 未激活羁绊
        units: 场上单位
        bench_units: 板凳单位
        augments: 强化符文
        hex_map: 海克斯地图 {(4,2):"blue_hex"}
        combo_meter: 连击数
        enemy_hp_vec: 敌方生命值向量
        fps: 帧率
        ping_ms: 延迟
        timestamp: 时间戳
    """

    side: Literal["self", "enemy"]
    stage: str  # "4-2"等
    phase_timer: Optional[int]
    rank: int
    hp: int
    gold: int
    level: int
    xp_progress: tuple[int, int]
    shop_odds: dict[str, float]
    shop_cards: list[str]
    traits: dict[str, int]
    inactive_traits: dict[str, int]
    units: list[Unit]
    bench_units: list[Unit]
    augments: list[str]
    hex_map: dict[tuple[int, int], str]  # {(4,2):"blue_hex"}
    combo_meter: Optional[int]
    enemy_hp_vec: list[int]
    fps: Optional[int]
    ping_ms: Optional[int]
    timestamp: float 