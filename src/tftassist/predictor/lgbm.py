"""实现基于LightGBM的预测功能。"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np

from ..core.state import BoardState

logger = logging.getLogger(__name__)

class LGBMPredictor:
    """LightGBM预测器类。"""
    
    def __init__(self, model_path: str) -> None:
        """初始化LightGBM预测器。
        
        Args:
            model_path: 模型路径
        """
        self.model = lgb.Booster(model_file=model_path)
        logger.info(f"加载 LightGBM 模型: {model_path}")
        
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """预测游戏状态。
        
        Args:
            features: 特征向量
            
        Returns:
            预测结果和特征重要性
        """
        # 预测
        pred = np.array(self.model.predict(features), dtype=np.float32)
        
        # 获取特征重要性
        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.model.feature_name()
        importance_dict = {name: float(imp) for name, imp in zip(feature_names, importance)}
        
        return pred, importance_dict
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """训练模型。
        
        Args:
            X: 特征矩阵
            y: 标签向量
            params: 模型参数
        """
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9
            }
        
        # 创建数据集
        train_data = lgb.Dataset(X, label=y)
        
        # 训练模型
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100
        )
        
        logger.info("模型训练完成") 