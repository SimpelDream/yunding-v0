"""数据检查脚本。"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tftassist.core.feature import build_feature
from tftassist.vision.detector import YOLONASDetector

logger = logging.getLogger(__name__)

def check_detector_data(
    data_dir: str,
    detector_path: str,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    min_objects: int = 1,
    max_objects: int = 100
) -> None:
    """检查检测器数据。
    
    Args:
        data_dir: 数据目录
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
    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"
    
    # 检查图像和标签
    invalid_images = []
    invalid_labels = []
    
    # 遍历图像
    for image_path in image_dir.glob("*.jpg"):
        # 检查图像是否可读
        try:
            import cv2
            img = cv2.imread(str(image_path))
            if img is None:
                invalid_images.append(image_path)
                logger.error(f"无法读取图像: {image_path}")
                continue
        except Exception as e:
            invalid_images.append(image_path)
            logger.error(f"读取图像失败: {image_path}, 错误: {e}")
            continue
        
        # 检测目标
        results = detector.detect(
            image_path=str(image_path),
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )
        
        # 检查目标数量
        if not (min_objects <= len(results) <= max_objects):
            invalid_images.append(image_path)
            logger.error(f"图像 {image_path.name} 的目标数量 {len(results)} 不在范围内 [{min_objects}, {max_objects}]")
            continue
        
        # 检查标签
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            invalid_labels.append(label_path)
            logger.error(f"缺少标签文件: {label_path}")
            continue
        
        # 验证标签格式
        try:
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        invalid_labels.append(label_path)
                        logger.error(f"标签格式错误: {label_path}, 行: {line}")
                        break
                    
                    class_id, x, y, w, h = map(float, parts)
                    if not (0 <= class_id < 100 and 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        invalid_labels.append(label_path)
                        logger.error(f"标签值超出范围: {label_path}, 行: {line}")
                        break
        except Exception as e:
            invalid_labels.append(label_path)
            logger.error(f"读取标签失败: {label_path}, 错误: {e}")
    
    # 输出检查结果
    logger.info(f"检测器数据检查完成:")
    logger.info(f"- 总图像数: {len(list(image_dir.glob('*.jpg')))}")
    logger.info(f"- 无效图像数: {len(invalid_images)}")
    logger.info(f"- 无效标签数: {len(invalid_labels)}")

def check_predictor_data(
    data_dir: str,
    min_units: int = 1,
    max_units: int = 9
) -> None:
    """检查预测器数据。
    
    Args:
        data_dir: 数据目录
        min_units: 最小单位数量
        max_units: 最大单位数量
    """
    # 准备数据
    data_dir = Path(data_dir)
    state_dir = data_dir / "states"
    
    # 检查状态
    invalid_states = []
    
    # 遍历状态文件
    for state_path in state_dir.glob("*.json"):
        try:
            import json
            with open(state_path) as f:
                state = json.load(f)
            
            # 检查必要字段
            required_fields = ["units", "features", "win_rate"]
            for field in required_fields:
                if field not in state:
                    invalid_states.append(state_path)
                    logger.error(f"状态缺少必要字段: {state_path}, 字段: {field}")
                    break
            
            # 检查单位数量
            if "units" in state:
                unit_count = len(state["units"])
                if not (min_units <= unit_count <= max_units):
                    invalid_states.append(state_path)
                    logger.error(f"状态 {state_path.name} 的单位数量 {unit_count} 不在范围内 [{min_units}, {max_units}]")
            
            # 检查特征
            if "features" in state:
                features = state["features"]
                if not isinstance(features, dict):
                    invalid_states.append(state_path)
                    logger.error(f"状态 {state_path.name} 的特征格式错误")
            
            # 检查胜率
            if "win_rate" in state:
                win_rate = state["win_rate"]
                if not (0 <= win_rate <= 1):
                    invalid_states.append(state_path)
                    logger.error(f"状态 {state_path.name} 的胜率 {win_rate} 不在范围内 [0, 1]")
        
        except Exception as e:
            invalid_states.append(state_path)
            logger.error(f"读取状态失败: {state_path}, 错误: {e}")
    
    # 输出检查结果
    logger.info(f"预测器数据检查完成:")
    logger.info(f"- 总状态数: {len(list(state_dir.glob('*.json')))}")
    logger.info(f"- 无效状态数: {len(invalid_states)}")

def main() -> None:
    """主函数。"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 检查检测器数据
    check_detector_data(
        data_dir="data/detector",
        detector_path="models/yolo_nas.pt",
        conf_thres=0.25,
        iou_thres=0.45,
        min_objects=1,
        max_objects=100
    )
    
    # 检查预测器数据
    check_predictor_data(
        data_dir="data/predictor",
        min_units=1,
        max_units=9
    )

if __name__ == "__main__":
    main() 