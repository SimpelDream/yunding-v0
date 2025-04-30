"""特征构建模块。

此模块实现了从棋盘状态构建特征向量的功能。
"""

import numpy as np
from numba import njit
from typing import Dict, List

from .state import BoardState, TRAIT_VOCAB

@njit(cache=True, nogil=True)
def _pad(vec: np.ndarray, length: int) -> np.ndarray:
    """填充向量到指定长度。

    Args:
        vec: 输入向量
        length: 目标长度

    Returns:
        填充后的向量
    """
    out = np.zeros(length, dtype=np.float32)
    n = min(len(vec), length)
    out[:n] = vec[:n]
    return out

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