"""数据导出脚本。"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tftassist.core.feature import build_feature
from tftassist.vision.detector import YOLONASDetector

logger = logging.getLogger(__name__)

def export_detector_data(
    data_dir: str,
    save_dir: str,
    format: str = "yolo"
) -> None:
    """导出检测器数据。
    
    Args:
        data_dir: 数据目录
        save_dir: 保存目录
        format: 导出格式，支持 "yolo" 和 "coco"
    """
    # 准备数据
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"
    
    # 创建目录
    (save_dir / "images").mkdir(parents=True, exist_ok=True)
    (save_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    if format == "yolo":
        # 遍历图像
        for image_path in image_dir.glob("*.jpg"):
            # 复制图像
            import shutil
            shutil.copy2(
                image_path,
                save_dir / "images" / image_path.name
            )
            
            # 复制标签
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(
                    label_path,
                    save_dir / "labels" / label_path.name
                )
    
    elif format == "coco":
        # 准备COCO格式数据
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # 遍历图像
        for image_id, image_path in enumerate(image_dir.glob("*.jpg")):
            # 复制图像
            import shutil
            shutil.copy2(
                image_path,
                save_dir / "images" / image_path.name
            )
            
            # 添加图像信息
            coco_data["images"].append({
                "id": image_id,
                "file_name": image_path.name,
                "width": 1920,  # TODO: 获取实际图像尺寸
                "height": 1080
            })
            
            # 添加标签信息
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                with open(label_path) as f:
                    for line in f:
                        class_id, x, y, w, h = map(float, line.strip().split())
                        coco_data["annotations"].append({
                            "id": len(coco_data["annotations"]),
                            "image_id": image_id,
                            "category_id": int(class_id),
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
        
        # 保存COCO格式数据
        import json
        with open(save_dir / "annotations.json", "w") as f:
            json.dump(coco_data, f)
    
    else:
        raise ValueError(f"不支持的导出格式: {format}")
    
    logger.info(f"检测器数据导出完成，格式: {format}")

def export_predictor_data(
    data_dir: str,
    save_dir: str,
    format: str = "csv"
) -> None:
    """导出预测器数据。
    
    Args:
        data_dir: 数据目录
        save_dir: 保存目录
        format: 导出格式，支持 "csv" 和 "json"
    """
    # 准备数据
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    state_dir = data_dir / "states"
    
    # 创建目录
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if format == "csv":
        # 准备CSV数据
        data = []
        
        # 遍历状态文件
        for state_path in state_dir.glob("*.json"):
            import json
            with open(state_path) as f:
                state = json.load(f)
            
            # 提取特征
            features = state.get("features", {})
            data.append(features)
        
        # 保存CSV数据
        df = pd.DataFrame(data)
        df.to_csv(save_dir / "data.csv", index=False)
    
    elif format == "json":
        # 遍历状态文件
        for state_path in state_dir.glob("*.json"):
            import shutil
            shutil.copy2(
                state_path,
                save_dir / state_path.name
            )
    
    else:
        raise ValueError(f"不支持的导出格式: {format}")
    
    logger.info(f"预测器数据导出完成，格式: {format}")

def main() -> None:
    """主函数。"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 导出检测器数据
    export_detector_data(
        data_dir="data",
        save_dir="data_exported",
        format="yolo"
    )
    
    # 导出预测器数据
    export_predictor_data(
        data_dir="data",
        save_dir="data_exported",
        format="csv"
    )

if __name__ == "__main__":
    main() 