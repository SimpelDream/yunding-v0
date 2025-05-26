"""LightGBM预测器模块。

此模块实现了基于LightGBM的胜率和伤害预测功能。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

from ..core.feature import build_feature
from ..core.state import BoardState

logger = logging.getLogger(__name__)

class LGBMPredictor:
    """LightGBM预测器。"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """初始化预测器。
        
        Args:
            model_path: 模型文件路径
            params: 模型参数
        """
        self.model: Optional[lgb.Booster] = None
        self.params = params or {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9
        }
        
        if model_path:
            self.load_model(model_path)
            
    def load_model(self, model_path: str) -> None:
        """加载模型。
        
        Args:
            model_path: 模型文件路径
        """
        try:
            self.model = lgb.Booster(model_file=model_path)
            logger.info(f"已加载模型: {model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
            
    def save_model(self, model_path: str) -> None:
        """保存模型。
        
        Args:
            model_path: 模型文件路径
        """
        if self.model is None:
            raise ValueError("模型未初始化")
            
        try:
            self.model.save_model(model_path)
            logger.info(f"已保存模型: {model_path}")
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            raise
            
    def predict(self, state: BoardState) -> float:
        """预测游戏状态。
        
        Args:
            state: 游戏状态
            
        Returns:
            预测值
        """
        if self.model is None:
            raise ValueError("模型未初始化")
            
        # 构建特征
        features = build_feature(state)
        
        # 预测
        prediction = self.model.predict([features])[0]
        return float(prediction)
        
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[float, float]:
        """训练模型。
        
        Args:
            X: 特征矩阵
            y: 目标值
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            训练集和测试集的RMSE
        """
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # 训练模型
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, test_data],
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        # 计算RMSE
        train_rmse = np.sqrt(np.mean((self.model.predict(X_train) - y_train) ** 2))
        test_rmse = np.sqrt(np.mean((self.model.predict(X_test) - y_test) ** 2))
        
        return train_rmse, test_rmse

    def build_feature(self, state: BoardState) -> np.ndarray:
        """构建特征向量。
        
        Args:
            state: 游戏状态
            
        Returns:
            特征向量
        """
        features = []
        
        # 基础特征
        features.extend([
            state.level,
            state.hp,
            state.gold,
            state.rank,
            len(state.units),
            len(state.bench_units)
        ])
        
        # 特质特征
        for trait, count in state.traits.items():
            features.append(count)
        for trait, count in state.inactive_traits.items():
            features.append(count)
            
        # 单位特征
        unit_features = self._extract_unit_features(state.units)
        features.extend(unit_features)
        
        # 备战区特征
        bench_features = self._extract_unit_features(state.bench_units)
        features.extend(bench_features)
        
        # 海克斯特征
        hex_features = self._extract_hex_features(state.hex_map)
        features.extend(hex_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_unit_features(self, units: List[Dict]) -> List[float]:
        """提取单位特征。
        
        Args:
            units: 单位列表
            
        Returns:
            单位特征列表
        """
        features = []
        
        # 单位数量
        features.append(len(units))
        
        # 星级统计
        star_counts = {1: 0, 2: 0, 3: 0}
        for unit in units:
            star_counts[unit["star"]] += 1
        features.extend(star_counts.values())
        
        # 装备统计
        item_counts = {}
        for unit in units:
            for item in unit["items"]:
                item_counts[item] = item_counts.get(item, 0) + 1
        features.extend(item_counts.values())
        
        return features
    
    def _extract_hex_features(self, hex_map: Dict[Tuple[int, int], str]) -> List[float]:
        """提取海克斯特征。
        
        Args:
            hex_map: 海克斯地图
            
        Returns:
            海克斯特征列表
        """
        features = []
        
        # 海克斯数量
        features.append(len(hex_map))
        
        # 海克斯类型统计
        hex_counts = {"blue_hex": 0, "artifact_hex": 0, "void_hex": 0}
        for hex_type in hex_map.values():
            hex_counts[hex_type] += 1
        features.extend(hex_counts.values())
        
        return features 