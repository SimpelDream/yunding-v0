"""数据准备脚本。"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tftassist.core.feature import build_feature
from tftassist.vision.detector import YOLONASDetector

logger = logging.getLogger(__name__)

def prepare_detector_data(
    data_dir: str,
    detector_path: str,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45
) -> None:
    """准备检测器数据。
    
    Args:
        data_dir: 数据目录
        detector_path: 检测器路径
        conf_thres: 置信度阈值
        iou_thres: IOU阈值
    """
    # 加载检测器
    detector = YOLONASDetector(model_path=detector_path)
    
    # 准备数据
    data_dir = Path(data_dir)
    image_dir = data_dir / "images"
    
    # 创建数据配置文件
    data_yaml = {
        "path": str(data_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {
            0: "unit",
            1: "item",
            2: "trait"
        }
    }
    
    # 保存配置文件
    import yaml
    with open(data_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)
    
    logger.info("检测器数据准备完成")

def prepare_predictor_data(
    data_dir: str,
    detector_path: str,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45
) -> None:
    """准备预测器数据。
    
    Args:
        data_dir: 数据目录
        detector_path: 检测器路径
        conf_thres: 置信度阈值
        iou_thres: IOU阈值
    """
    # 加载检测器
    detector = YOLONASDetector(model_path=detector_path)
    
    # 准备数据
    data_dir = Path(data_dir)
    image_dir = data_dir / "images"
    state_dir = data_dir / "states"
    
    # 遍历图像
    for image_path in image_dir.glob("*.jpg"):
        # 检测目标
        results = detector.detect(
            image_path=str(image_path),
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )
        
        # 构建特征
        features = build_feature(results)
        
        # 保存特征
        feature_path = state_dir / f"{image_path.stem}.json"
        import json
        with open(feature_path, "w") as f:
            json.dump(features, f)
    
    logger.info("预测器数据准备完成")

def main() -> None:
    """主函数。"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 准备检测器数据
    prepare_detector_data(
        data_dir="data",
        detector_path="models/yolo_nas.pt",
        conf_thres=0.25,
        iou_thres=0.45
    )
    
    # 准备预测器数据
    prepare_predictor_data(
        data_dir="data",
        detector_path="models/yolo_nas.pt",
        conf_thres=0.25,
        iou_thres=0.45
    )

if __name__ == "__main__":
    main() 