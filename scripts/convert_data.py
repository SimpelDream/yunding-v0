"""数据转换脚本。"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tftassist.core.feature import build_feature
from tftassist.vision.detector import YOLONASDetector

logger = logging.getLogger(__name__)

def convert_detector_data(
    data_dir: str,
    save_dir: str,
    format: str = "yolo"
) -> None:
    """转换检测器数据。
    
    Args:
        data_dir: 数据目录
        save_dir: 保存目录
        format: 目标格式，支持 "yolo"
    """
    # 准备数据
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    
    # 创建保存目录
    save_dir.mkdir(parents=True, exist_ok=True)
    image_dir = save_dir / "images"
    label_dir = save_dir / "labels"
    image_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)
    
    # 复制图像
    for image_path in data_dir.glob("*.jpg"):
        import shutil
        shutil.copy2(image_path, image_dir / image_path.name)
    
    # 转换标签
    if format == "yolo":
        # 复制标签
        for label_path in data_dir.glob("*.txt"):
            import shutil
            shutil.copy2(label_path, label_dir / label_path.name)
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    logger.info(f"检测器数据转换完成: {format}")

def convert_predictor_data(
    data_dir: str,
    save_dir: str,
    format: str = "csv"
) -> None:
    """转换预测器数据。
    
    Args:
        data_dir: 数据目录
        save_dir: 保存目录
        format: 目标格式，支持 "csv"
    """
    # 准备数据
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    
    # 创建保存目录
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载状态数据
    states = []
    for state_path in data_dir.glob("*.json"):
        import json
        with open(state_path) as f:
            state = json.load(f)
            states.append(state)
    
    # 构建特征矩阵
    features = []
    for state in states:
        feature = build_feature(state)
        features.append(feature)
    
    # 转换数据
    if format == "csv":
        # 保存为CSV
        df = pd.DataFrame(features)
        df.to_csv(save_dir / "features.csv", index=False)
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    logger.info(f"预测器数据转换完成: {format}")

def main() -> None:
    """主函数。"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 转换检测器数据
    convert_detector_data(
        data_dir="data/detector",
        save_dir="data/detector_converted",
        format="yolo"
    )
    
    # 转换预测器数据
    convert_predictor_data(
        data_dir="data/predictor",
        save_dir="data/predictor_converted",
        format="csv"
    )

if __name__ == "__main__":
    main() 