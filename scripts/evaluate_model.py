"""模型评估脚本。"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

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

def evaluate_detector(
    data_dir: str,
    model_path: str,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45
) -> Dict[str, float]:
    """评估检测器。
    
    Args:
        data_dir: 数据目录
        model_path: 模型路径
        conf_thres: 置信度阈值
        iou_thres: IOU阈值
        
    Returns:
        评估指标
    """
    # 加载模型
    detector = YOLONASDetector(model_path=model_path)
    
    # 准备数据
    data_yaml = Path(data_dir) / "data.yaml"
    
    # 评估模型
    metrics = detector.evaluate(
        data_yaml=str(data_yaml),
        conf_thres=conf_thres,
        iou_thres=iou_thres
    )
    
    logger.info(f"检测器评估完成: {metrics}")
    return metrics

def evaluate_predictor(
    data_dir: str,
    model_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, float]:
    """评估预测器。
    
    Args:
        data_dir: 数据目录
        model_path: 模型路径
        test_size: 测试集比例
        random_state: 随机种子
        
    Returns:
        评估指标
    """
    # 加载数据
    X, y = load_data(data_dir)
    
    # 创建预测器
    predictor = LGBMPredictor()
    predictor.load_model(model_path)
    
    # 评估模型
    y_pred = predictor.predict(X)
    
    # 计算指标
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "r2": r2_score(y, y_pred)
    }
    
    logger.info(f"预测器评估完成: {metrics}")
    return metrics

def main() -> None:
    """主函数。"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 评估检测器
    detector_metrics = evaluate_detector(
        data_dir="data",
        model_path="models/yolo_nas.pt",
        conf_thres=0.25,
        iou_thres=0.45
    )
    
    # 评估预测器
    predictor_metrics = evaluate_predictor(
        data_dir="data",
        model_path="models/lgbm.txt",
        test_size=0.2,
        random_state=42
    )

if __name__ == "__main__":
    main() 