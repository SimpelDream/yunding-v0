"""模型导出脚本。"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from tftassist.predictor.lgbm import LGBMPredictor
from tftassist.vision.detector import YOLONASDetector

logger = logging.getLogger(__name__)

def export_detector(
    model_path: str,
    save_dir: str,
    format: str = "onnx"
) -> None:
    """导出检测器。
    
    Args:
        model_path: 模型路径
        save_dir: 保存目录
        format: 导出格式
    """
    # 加载模型
    detector = YOLONASDetector(model_path=model_path)
    
    # 导出模型
    save_path = Path(save_dir) / f"detector.{format}"
    detector.export(save_path=str(save_path), format=format)
    
    logger.info(f"检测器已导出至: {save_path}")

def export_predictor(
    model_path: str,
    save_dir: str,
    format: str = "onnx"
) -> None:
    """导出预测器。
    
    Args:
        model_path: 模型路径
        save_dir: 保存目录
        format: 导出格式
    """
    # 加载模型
    predictor = LGBMPredictor()
    predictor.load_model(model_path)
    
    # 导出模型
    save_path = Path(save_dir) / f"predictor.{format}"
    predictor.export(save_path=str(save_path), format=format)
    
    logger.info(f"预测器已导出至: {save_path}")

def main() -> None:
    """主函数。"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 导出检测器
    export_detector(
        model_path="models/yolo_nas.pt",
        save_dir="models",
        format="onnx"
    )
    
    # 导出预测器
    export_predictor(
        model_path="models/lgbm.txt",
        save_dir="models",
        format="onnx"
    )

if __name__ == "__main__":
    main() 