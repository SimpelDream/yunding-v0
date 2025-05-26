"""数据清理脚本。"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tftassist.core.feature import build_feature
from tftassist.vision.detector import YOLONASDetector

logger = logging.getLogger(__name__)

def clean_detector_data(
    data_dir: str,
    save_dir: str,
    detector_path: str,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    min_objects: int = 1,
    max_objects: int = 100
) -> None:
    """清理检测器数据。
    
    Args:
        data_dir: 数据目录
        save_dir: 保存目录
        detector_path: 检测器路径
        conf_thres: 置信度阈值
        iou_thres: IOU阈值
        min_objects: 最小目标数量
        max_objects: 最大目标数量
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
        
        # 检查目标数量
        if min_objects <= len(results) <= max_objects:
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
        else:
            logger.warning(f"图像 {image_path.name} 的目标数量 {len(results)} 不在范围内 [{min_objects}, {max_objects}]")
    
    logger.info("检测器数据清理完成")

def clean_predictor_data(
    data_dir: str,
    save_dir: str,
    detector_path: str,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    min_objects: int = 1,
    max_objects: int = 100
) -> None:
    """清理预测器数据。
    
    Args:
        data_dir: 数据目录
        save_dir: 保存目录
        detector_path: 检测器路径
        conf_thres: 置信度阈值
        iou_thres: IOU阈值
        min_objects: 最小目标数量
        max_objects: 最大目标数量
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
        
        # 检查目标数量
        if min_objects <= len(results) <= max_objects:
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
        else:
            logger.warning(f"图像 {image_path.name} 的目标数量 {len(results)} 不在范围内 [{min_objects}, {max_objects}]")
    
    logger.info("预测器数据清理完成")

def main() -> None:
    """主函数。"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 清理检测器数据
    clean_detector_data(
        data_dir="data",
        save_dir="data_cleaned",
        detector_path="models/yolo_nas.pt",
        conf_thres=0.25,
        iou_thres=0.45,
        min_objects=1,
        max_objects=100
    )
    
    # 清理预测器数据
    clean_predictor_data(
        data_dir="data",
        save_dir="data_cleaned",
        detector_path="models/yolo_nas.pt",
        conf_thres=0.25,
        iou_thres=0.45,
        min_objects=1,
        max_objects=100
    )

if __name__ == "__main__":
    main() 