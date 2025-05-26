"""数据处理脚本。"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tftassist.core.feature import build_feature
from tftassist.vision.detector import YOLONASDetector

logger = logging.getLogger(__name__)

def process_detector_data(
    data_dir: str,
    save_dir: str,
    detector_path: str,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45
) -> None:
    """处理检测器数据。
    
    Args:
        data_dir: 数据目录
        save_dir: 保存目录
        detector_path: 检测器路径
        conf_thres: 置信度阈值
        iou_thres: IOU阈值
    """
    # 加载检测器
    detector = YOLONASDetector(model_path=detector_path)
    
    # 准备数据
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    image_dir = data_dir / "images"
    
    # 创建目录
    (save_dir / "images").mkdir(parents=True, exist_ok=True)
    (save_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    # 遍历图像
    for image_path in image_dir.glob("*.jpg"):
        # 检测目标
        results = detector.detect(
            image_path=str(image_path),
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )
        
        if results:
            # 保存图像
            import shutil
            shutil.copy2(
                image_path,
                save_dir / "images" / image_path.name
            )
            
            # 保存标签
            label_path = save_dir / "labels" / f"{image_path.stem}.txt"
            with open(label_path, "w") as f:
                for result in results:
                    f.write(f"{result['class']} {result['bbox'][0]} {result['bbox'][1]} {result['bbox'][2]} {result['bbox'][3]}\n")
    
    logger.info("检测器数据处理完成")

def process_predictor_data(
    data_dir: str,
    save_dir: str,
    detector_path: str,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45
) -> None:
    """处理预测器数据。
    
    Args:
        data_dir: 数据目录
        save_dir: 保存目录
        detector_path: 检测器路径
        conf_thres: 置信度阈值
        iou_thres: IOU阈值
    """
    # 加载检测器
    detector = YOLONASDetector(model_path=detector_path)
    
    # 准备数据
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    image_dir = data_dir / "images"
    state_dir = data_dir / "states"
    
    # 创建目录
    (save_dir / "states").mkdir(parents=True, exist_ok=True)
    
    # 遍历图像
    for image_path in image_dir.glob("*.jpg"):
        # 检测目标
        results = detector.detect(
            image_path=str(image_path),
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )
        
        if results:
            # 构建特征
            features = build_feature(results)
            
            # 加载状态
            state_path = state_dir / f"{image_path.stem}.json"
            if state_path.exists():
                import json
                with open(state_path) as f:
                    state = json.load(f)
                
                # 更新状态
                state["features"] = features
                
                # 保存状态
                save_path = save_dir / "states" / state_path.name
                with open(save_path, "w") as f:
                    json.dump(state, f)
    
    logger.info("预测器数据处理完成")

def main() -> None:
    """主函数。"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 处理检测器数据
    process_detector_data(
        data_dir="data",
        save_dir="data_processed",
        detector_path="models/yolo_nas.pt",
        conf_thres=0.25,
        iou_thres=0.45
    )
    
    # 处理预测器数据
    process_predictor_data(
        data_dir="data",
        save_dir="data_processed",
        detector_path="models/yolo_nas.pt",
        conf_thres=0.25,
        iou_thres=0.45
    )

if __name__ == "__main__":
    main() 