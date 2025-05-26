"""特征工程模块。

此模块实现了游戏状态的特征提取功能。
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from numba import njit

from .state import BoardState, Unit, TRAIT_VOCAB

logger = logging.getLogger(__name__)

@njit(cache=True)
def _extract_traits(units: List[Unit]) -> np.ndarray:
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

def extract_features(state: BoardState) -> np.ndarray:
    """提取游戏状态特征。
    
    Args:
        state: 游戏状态
        
    Returns:
        特征向量
    """
    # 提取特质特征
    trait_vec = _extract_traits(state.units)
    
    # 提取其他特征
    other_features = np.array([
        float(state.stage),
        state.phase_timer,
        float(state.rank),
        float(state.hp),
        float(state.gold),
        float(state.level),
        state.xp_progress,
        float(state.board_height),
        float(state.board_width),
        state.hp_pct,
        state.shield_pct
    ], dtype=np.float32)
    
    # 合并特征
    features = np.concatenate([trait_vec, other_features]).astype(np.float32)
    
    return features

def _pad(x: np.ndarray, size: int) -> np.ndarray:
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

def build_feature(state: BoardState) -> np.ndarray:
    """将棋盘状态转换为机器学习特征向量。

    Args:
        state: 棋盘状态对象

    Returns:
        特征向量，包含以下部分：
        - one-hot 羁绊特征
        - 板凳羁绊特征
        - 连续特征（生命值百分比、护盾百分比、金币、等级）
        - 敌方生命值向量（填充到7个）
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
    cont = np.array(
        [
            state.hp_pct,
            state.shield_pct,
            state.gold,
            state.level,
        ],
        dtype=np.float32,
    )
    
    # 4. 敌方生命值向量
    enemy = _pad(np.asarray(state.enemy_hp_vec, dtype=np.float32), 7)
    
    # 拼接所有特征
    return np.concatenate([trait_vec, bench_vec, cont, enemy]) 