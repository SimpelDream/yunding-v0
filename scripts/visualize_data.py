"""数据可视化脚本。"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tftassist.core.feature import build_feature
from tftassist.vision.detector import YOLONASDetector

logger = logging.getLogger(__name__)

def visualize_detector_data(
    data_dir: str,
    save_dir: str,
    detector_path: str,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45
) -> None:
    """可视化检测器数据。
    
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
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备统计数据
    class_counts = {}
    bbox_sizes = []
    confidences = []
    
    # 遍历图像
    for image_path in image_dir.glob("*.jpg"):
        # 检测目标
        results = detector.detect(
            image_path=str(image_path),
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )
        
        # 统计类别
        for result in results:
            class_id = result["class"]
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            
            # 统计边界框尺寸
            bbox = result["bbox"]
            bbox_sizes.append(bbox[2] * bbox[3])  # 宽 * 高
            
            # 统计置信度
            confidences.append(result["confidence"])
    
    # 绘制类别分布
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title("类别分布")
    plt.xlabel("类别")
    plt.ylabel("数量")
    plt.savefig(save_dir / "class_distribution.png")
    plt.close()
    
    # 绘制边界框尺寸分布
    plt.figure(figsize=(10, 6))
    plt.hist(bbox_sizes, bins=50)
    plt.title("边界框尺寸分布")
    plt.xlabel("尺寸")
    plt.ylabel("数量")
    plt.savefig(save_dir / "bbox_size_distribution.png")
    plt.close()
    
    # 绘制置信度分布
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=50)
    plt.title("置信度分布")
    plt.xlabel("置信度")
    plt.ylabel("数量")
    plt.savefig(save_dir / "confidence_distribution.png")
    plt.close()
    
    logger.info("检测器数据可视化完成")

def visualize_predictor_data(
    data_dir: str,
    save_dir: str
) -> None:
    """可视化预测器数据。
    
    Args:
        data_dir: 数据目录
        save_dir: 保存目录
    """
    # 准备数据
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    state_dir = data_dir / "states"
    
    # 创建目录
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备数据
    data = []
    
    # 遍历状态文件
    for state_path in state_dir.glob("*.json"):
        import json
        with open(state_path) as f:
            state = json.load(f)
        
        # 提取特征
        features = state.get("features", {})
        data.append(features)
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 绘制特征分布
    plt.figure(figsize=(15, 10))
    sns.pairplot(df)
    plt.savefig(save_dir / "feature_distribution.png")
    plt.close()
    
    # 绘制特征相关性
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("特征相关性")
    plt.savefig(save_dir / "feature_correlation.png")
    plt.close()
    
    logger.info("预测器数据可视化完成")

def main() -> None:
    """主函数。"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 可视化检测器数据
    visualize_detector_data(
        data_dir="data",
        save_dir="data_visualized",
        detector_path="models/yolo_nas.pt",
        conf_thres=0.25,
        iou_thres=0.45
    )
    
    # 可视化预测器数据
    visualize_predictor_data(
        data_dir="data",
        save_dir="data_visualized"
    )

if __name__ == "__main__":
    main() 