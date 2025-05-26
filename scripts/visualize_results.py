"""结果可视化脚本。"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str
) -> None:
    """绘制混淆矩阵。
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        save_path: 保存路径
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"混淆矩阵已保存至: {save_path}")

def plot_feature_importance(
    model_path: str,
    save_path: str
) -> None:
    """绘制特征重要性。
    
    Args:
        model_path: 模型路径
        save_path: 保存路径
    """
    # 加载模型
    predictor = LGBMPredictor()
    predictor.load_model(model_path)
    
    # 获取特征重要性
    importance = predictor.get_feature_importance()
    
    # 绘制条形图
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importance)), importance)
    plt.title("特征重要性")
    plt.xlabel("特征索引")
    plt.ylabel("重要性")
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"特征重要性图已保存至: {save_path}")

def plot_prediction_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str
) -> None:
    """绘制预测误差。
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        save_path: 保存路径
    """
    # 计算误差
    errors = y_true - y_pred
    
    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, errors)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.title("预测误差分布")
    plt.xlabel("真实值")
    plt.ylabel("误差")
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"预测误差图已保存至: {save_path}")

def main() -> None:
    """主函数。"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 加载数据
    X, y = load_data("data")
    
    # 加载模型
    predictor = LGBMPredictor()
    predictor.load_model("models/lgbm.txt")
    
    # 获取预测结果
    y_pred = predictor.predict(X)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        y_true=y,
        y_pred=y_pred,
        save_path="results/confusion_matrix.png"
    )
    
    # 绘制特征重要性
    plot_feature_importance(
        model_path="models/lgbm.txt",
        save_path="results/feature_importance.png"
    )
    
    # 绘制预测误差
    plot_prediction_error(
        y_true=y,
        y_pred=y_pred,
        save_path="results/prediction_error.png"
    )

if __name__ == "__main__":
    main() 