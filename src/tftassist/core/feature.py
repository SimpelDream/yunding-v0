"""特征工程模块。

此模块实现了游戏状态特征提取和构建功能。
"""

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .state import BoardState, Unit, TRAIT_VOCAB

logger = logging.getLogger(__name__)

@njit(cache=True)
def _extract_traits(units: List[Unit]) -> NDArray[np.float32]:
    """提取特质特征。
    
    Args:
        units: 单位列表
        
    Returns:
        特质特征向量
    """
    trait_list = list(TRAIT_VOCAB.keys())
    trait_vec = np.zeros(len(trait_list), dtype=np.float32)
    
    for unit in units:
        for trait in unit.traits:
            if trait in trait_list:
                idx = trait_list.index(trait)
                trait_vec[idx] += 1
                
    return trait_vec

@njit(cache=True)
def _extract_hex_counts(state: BoardState) -> NDArray[np.float32]:
    """提取六边形数量特征。
    
    Args:
        state: 游戏状态
        
    Returns:
        六边形数量向量 [blue_cnt, artifact_cnt, void_cnt]
    """
    hex_counts = np.zeros(3, dtype=np.float32)
    for hex_type in state.hex_map.values():
        if hex_type == "blue_hex":
            hex_counts[0] += 1
        elif hex_type == "artifact_hex":
            hex_counts[1] += 1
        elif hex_type == "void_hex":
            hex_counts[2] += 1
    return hex_counts

@njit(cache=True)
def _normalize_stage(stage: str) -> float:
    """将阶段字符串转换为浮点数。
    
    Args:
        stage: 阶段字符串, 如 "4-2"
        
    Returns:
        浮点数, 如 4.2
    """
    try:
        major, minor = stage.split("-")
        return float(major) + float(minor) / 10.0
    except (ValueError, AttributeError):
        return 0.0

@njit(cache=True)
def _normalize_xp(current: int, total: int) -> float:
    """归一化经验值进度。
    
    Args:
        current: 当前经验值
        total: 总经验值
        
    Returns:
        归一化后的进度, 范围 [0, 1]
    """
    return float(current) / float(total) if total > 0 else 0.0

@njit(cache=True)
def _pad_array(x: NDArray[np.float32], size: int) -> NDArray[np.float32]:
    """填充数组到指定大小。
    
    Args:
        x: 输入数组
        size: 目标大小
        
    Returns:
        填充后的数组
    """
    if len(x) >= size:
        return x[:size]
    return np.pad(x, (0, size - len(x)), mode='constant')

@njit(cache=True)
def build_feature(state: BoardState) -> NDArray[np.float32]:
    """将棋盘状态转换为机器学习特征向量。

    Args:
        state: 棋盘状态对象

    Returns:
        特征向量, 包含以下部分:
        - one-hot 羁绊特征
        - 板凳羁绊特征
        - 连续特征(生命值百分比、护盾百分比、金币、等级)
        - 敌方生命值向量(填充到7个)
        - 六边形数量向量
    """
    # 1. one-hot 羁绊特征
    trait_vec = np.zeros(len(TRAIT_VOCAB), dtype=np.float32)
    for t, c in state.traits.items():
        trait_vec[TRAIT_VOCAB[t]] = c
        
    # 2. 板凳羁绊特征
    bench_vec = np.zeros(len(TRAIT_VOCAB), dtype=np.float32)
    for u in state.bench_units:
        for t in u.traits:
            bench_vec[TRAIT_VOCAB[t]] += 1
            
    # 3. 连续特征
    cont = np.array([
        state.hp_pct,
        state.shield_pct,
        state.gold,
        state.level,
        _normalize_stage(state.stage),
        _normalize_xp(state.xp_progress[0], state.xp_progress[1])
    ], dtype=np.float32)
    
    # 4. 敌方生命值向量
    enemy = _pad_array(np.asarray(state.enemy_hp_vec, dtype=np.float32), 7)
    
    # 5. 六边形数量向量
    hex_counts = _extract_hex_counts(state)
    
    # 拼接所有特征
    return np.concatenate([trait_vec, bench_vec, cont, enemy, hex_counts])

def extract_trait_features(
    traits: Dict[str, int],
    inactive_traits: Dict[str, int]
) -> List[float]:
    """提取特质特征。
    
    Args:
        traits: 激活的特质
        inactive_traits: 未激活的特质
        
    Returns:
        特质特征向量
    """
    features: List[float] = []
    
    # 激活的特质数量
    features.append(float(len(traits)))
    
    # 未激活的特质数量
    features.append(float(len(inactive_traits)))
    
    # 特质等级
    for trait in ["Mage", "Assassin", "Tank", "Support", "Marksman", "Fighter"]:
        features.append(float(traits.get(trait, 0)))
        features.append(float(inactive_traits.get(trait, 0)))
    
    return features

def extract_unit_features(units: Sequence[Unit]) -> List[float]:
    """提取单位特征。
    
    Args:
        units: 单位列表
        
    Returns:
        单位特征向量
    """
    features: List[float] = []
    
    # 单位数量
    features.append(float(len(units)))
    
    # 星级统计
    star_counts = [0, 0, 0]
    for unit in units:
        star_counts[unit.star - 1] += 1
    features.extend([float(count) for count in star_counts])
    
    # 装备数量
    item_counts = [0, 0, 0]
    for unit in units:
        item_counts[len(unit.items)] += 1
    features.extend([float(count) for count in item_counts])
    
    return features

def extract_hex_features(hex_map: Dict[Tuple[int, int], str]) -> List[float]:
    """提取六边形特征。
    
    Args:
        hex_map: 六边形地图
        
    Returns:
        六边形特征向量
    """
    features: List[float] = []
    
    # 六边形数量
    features.append(float(len(hex_map)))
    
    # 六边形类型统计
    hex_counts = {"Mage": 0, "Assassin": 0, "Tank": 0, "Support": 0, "Marksman": 0, "Fighter": 0}
    for hex_type in hex_map.values():
        hex_counts[hex_type] += 1
    features.extend([float(count) for count in hex_counts.values()])
    
    return features

def extract_hp_shield_features(units: Sequence[Unit]) -> List[float]:
    """提取生命值和护盾特征。
    
    Args:
        units: 单位列表
        
    Returns:
        生命值和护盾特征向量
    """
    features: List[float] = []
    
    # 平均生命值百分比
    if units:
        avg_hp = sum(unit.hp_pct for unit in units) / len(units)
        features.append(avg_hp)
    else:
        features.append(0.0)
    
    # 平均护盾百分比
    if units:
        avg_shield = sum(unit.shield_pct for unit in units) / len(units)
        features.append(avg_shield)
    else:
        features.append(0.0)
    
    return features 