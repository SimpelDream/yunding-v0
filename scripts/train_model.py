"""模型训练脚本。"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tftassist.core.feature import build_feature
from tftassist.predictor.lgbm import LGBMPredictor
from tftassist.vision.detector import YOLONASDetector

logger = logging.getLogger(__name__)

def load_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载数据。
    
    Args:
        data_dir: 数据目录
        
    Returns:
        特征矩阵和目标值
    """
    data_dir = Path(data_dir)
    
    # 加载状态数据
    states = []
    for state_file in (data_dir / "states").glob("*.json"):
        # TODO: 实现状态加载
        pass
        
    # 构建特征
    X = np.array([build_feature(state) for state in states])
    
    # 构建目标值
    y = np.array([state.rank for state in states])
    
    return X, y

def train_detector(
    data_dir: str,
    save_dir: str,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640
) -> None:
    """训练检测器。
    
    Args:
        data_dir: 数据目录
        save_dir: 保存目录
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 图像大小
    """
    # 准备数据
    data_yaml = Path(data_dir) / "data.yaml"
    
    # 训练模型
    YOLONASDetector.train(
        data_yaml=str(data_yaml),
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size
    )
    
    logger.info("检测器训练完成")

def train_predictor(
    data_dir: str,
    save_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> None:
    """训练预测器。
    
    Args:
        data_dir: 数据目录
        save_dir: 保存目录
        test_size: 测试集比例
        random_state: 随机种子
    """
    # 加载数据
    X, y = load_data(data_dir)
    
    # 创建预测器
    predictor = LGBMPredictor()
    
    # 训练模型
    train_rmse, test_rmse = predictor.train(
        X=X,
        y=y,
        test_size=test_size,
        random_state=random_state
    )
    
    # 保存模型
    save_path = Path(save_dir) / "lgbm.txt"
    predictor.save_model(str(save_path))
    
    logger.info(f"预测器训练完成, 训练集RMSE: {train_rmse:.4f}, 测试集RMSE: {test_rmse:.4f}")

def main() -> None:
    """主函数。"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 训练检测器
    train_detector(
        data_dir="data",
        save_dir="models",
        epochs=100,
        batch_size=16,
        img_size=640
    )
    
    # 训练预测器
    train_predictor(
        data_dir="data",
        save_dir="models",
        test_size=0.2,
        random_state=42
    )

if __name__ == "__main__":
    main() 