"""数据合并脚本。"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def merge_detector_data(
    data_dirs: List[str],
    save_dir: str
) -> None:
    """合并检测器数据。
    
    Args:
        data_dirs: 数据目录列表
        save_dir: 保存目录
    """
    # 准备数据
    data_dirs = [Path(d) for d in data_dirs]
    save_dir = Path(save_dir)
    
    # 创建保存目录
    save_dir.mkdir(parents=True, exist_ok=True)
    image_dir = save_dir / "images"
    label_dir = save_dir / "labels"
    image_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)
    
    # 合并数据
    for data_dir in data_dirs:
        # 复制图像
        for image_path in data_dir.glob("*.jpg"):
            import shutil
            shutil.copy2(image_path, image_dir / image_path.name)
        
        # 复制标签
        for label_path in data_dir.glob("*.txt"):
            import shutil
            shutil.copy2(label_path, label_dir / label_path.name)
    
    logger.info(f"检测器数据合并完成: {len(data_dirs)} 个目录")

def merge_predictor_data(
    data_dirs: List[str],
    save_dir: str
) -> None:
    """合并预测器数据。
    
    Args:
        data_dirs: 数据目录列表
        save_dir: 保存目录
    """
    # 准备数据
    data_dirs = [Path(d) for d in data_dirs]
    save_dir = Path(save_dir)
    
    # 创建保存目录
    save_dir.mkdir(parents=True, exist_ok=True)
    state_dir = save_dir / "states"
    state_dir.mkdir(exist_ok=True)
    
    # 合并数据
    for data_dir in data_dirs:
        # 复制状态文件
        for state_path in data_dir.glob("*.json"):
            import shutil
            shutil.copy2(state_path, state_dir / state_path.name)
    
    logger.info(f"预测器数据合并完成: {len(data_dirs)} 个目录")

def main() -> None:
    """主函数。"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 合并检测器数据
    merge_detector_data(
        data_dirs=[
            "data/detector_1",
            "data/detector_2"
        ],
        save_dir="data/detector_merged"
    )
    
    # 合并预测器数据
    merge_predictor_data(
        data_dirs=[
            "data/predictor_1",
            "data/predictor_2"
        ],
        save_dir="data/predictor_merged"
    )

if __name__ == "__main__":
    main() 