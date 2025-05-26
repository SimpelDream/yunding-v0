"""模型基准测试脚本。"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from tftassist.predictor.lgbm import LGBMPredictor
from tftassist.vision.detector import YOLONASDetector

logger = logging.getLogger(__name__)

def benchmark_detector(
    model_path: str,
    data_dir: str,
    batch_size: int = 1,
    num_runs: int = 100
) -> Dict[str, float]:
    """基准测试检测器。
    
    Args:
        model_path: 模型路径
        data_dir: 数据目录
        batch_size: 批次大小
        num_runs: 运行次数
        
    Returns:
        性能指标
    """
    # 加载模型
    detector = YOLONASDetector(model_path=model_path)
    
    # 准备数据
    data_yaml = Path(data_dir) / "data.yaml"
    
    # 预热
    detector.evaluate(
        data_yaml=str(data_yaml),
        conf_thres=0.25,
        iou_thres=0.45
    )
    
    # 基准测试
    start_time = time.time()
    for _ in range(num_runs):
        detector.evaluate(
            data_yaml=str(data_yaml),
            conf_thres=0.25,
            iou_thres=0.45
        )
    end_time = time.time()
    
    # 计算指标
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    fps = num_runs / total_time
    
    metrics = {
        "total_time": total_time,
        "avg_time": avg_time,
        "fps": fps
    }
    
    logger.info(f"检测器基准测试完成: {metrics}")
    return metrics

def benchmark_predictor(
    model_path: str,
    data_dir: str,
    batch_size: int = 1,
    num_runs: int = 100
) -> Dict[str, float]:
    """基准测试预测器。
    
    Args:
        model_path: 模型路径
        data_dir: 数据目录
        batch_size: 批次大小
        num_runs: 运行次数
        
    Returns:
        性能指标
    """
    # 加载模型
    predictor = LGBMPredictor()
    predictor.load_model(model_path)
    
    # 准备数据
    X = np.random.randn(num_runs, 100)  # TODO: 使用真实数据
    
    # 预热
    predictor.predict(X[:1])
    
    # 基准测试
    start_time = time.time()
    for _ in range(num_runs):
        predictor.predict(X[:1])
    end_time = time.time()
    
    # 计算指标
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    fps = num_runs / total_time
    
    metrics = {
        "total_time": total_time,
        "avg_time": avg_time,
        "fps": fps
    }
    
    logger.info(f"预测器基准测试完成: {metrics}")
    return metrics

def main() -> None:
    """主函数。"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 基准测试检测器
    detector_metrics = benchmark_detector(
        model_path="models/yolo_nas.pt",
        data_dir="data",
        batch_size=1,
        num_runs=100
    )
    
    # 基准测试预测器
    predictor_metrics = benchmark_predictor(
        model_path="models/lgbm.txt",
        data_dir="data",
        batch_size=1,
        num_runs=100
    )

if __name__ == "__main__":
    main() 