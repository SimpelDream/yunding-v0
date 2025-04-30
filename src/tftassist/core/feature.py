"""特征构建模块。

此模块实现了从棋盘状态构建特征向量的功能。
"""

import numpy as np
from numba import jit
from typing import Dict, List

@jit(nopython=True, cache=True)
def build_feature(state: 'BoardState') -> np.ndarray:
    """构建特征向量。

    Args:
        state: 棋盘状态对象

    Returns:
        特征向量，包含以下部分：
        - one-hot 羁绊特征
        - 板凳羁绊特征
        - 连续特征（生命值百分比、护盾百分比、金币、等级）
        - 敌方生命值向量（填充到7个）

    Note:
        使用 numba.jit 加速计算
    """
    # 初始化特征列表
    features = []
    
    # 1. one-hot 羁绊特征
    trait_counts = np.zeros(len(state.traits))
    for trait in state.active_traits:
        trait_counts[state.traits.index(trait)] = 1
    features.extend(trait_counts)
    
    # 2. 板凳羁绊特征
    bench_traits = np.zeros(len(state.traits))
    for unit in state.bench:
        for trait in unit.traits:
            bench_traits[state.traits.index(trait)] += 1
    features.extend(bench_traits)
    
    # 3. 连续特征
    hp_pct = state.player_hp / state.max_hp
    shield_pct = state.player_shield / state.max_shield if state.max_shield > 0 else 0
    features.extend([hp_pct, shield_pct, state.gold, state.level])
    
    # 4. 敌方生命值向量
    enemy_hp = np.array([p.hp for p in state.enemies])
    if len(enemy_hp) < 7:
        enemy_hp = np.pad(enemy_hp, (0, 7 - len(enemy_hp)))
    features.extend(enemy_hp)
    
    return np.array(features, dtype=np.float32) 