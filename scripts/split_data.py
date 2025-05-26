"""数据分割脚本。"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def split_detector_data(
    data_dir: str,
    save_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> None:
    """分割检测器数据。
    
    Args:
        data_dir: 数据目录
        save_dir: 保存目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子
    """
    # 准备数据
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    
    # 创建保存目录
    save_dir.mkdir(parents=True, exist_ok=True)
    train_dir = save_dir / "train"
    val_dir = save_dir / "val"
    test_dir = save_dir / "test"
    
    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(exist_ok=True)
        (split_dir / "images").mkdir(exist_ok=True)
        (split_dir / "labels").mkdir(exist_ok=True)
    
    # 获取图像列表
    image_paths = list(data_dir.glob("*.jpg"))
    
    # 分割数据
    train_paths, temp_paths = train_test_split(
        image_paths,
        train_size=train_ratio,
        random_state=random_state
    )
    
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths = train_test_split(
        temp_paths,
        train_size=val_ratio_adjusted,
        random_state=random_state
    )
    
    # 复制数据
    for split_name, split_paths in [
        ("train", train_paths),
        ("val", val_paths),
        ("test", test_paths)
    ]:
        split_dir = save_dir / split_name
        
        for image_path in split_paths:
            # 复制图像
            import shutil
            shutil.copy2(
                image_path,
                split_dir / "images" / image_path.name
            )
            
            # 复制标签
            label_path = data_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(
                    label_path,
                    split_dir / "labels" / label_path.name
                )
    
    logger.info(f"检测器数据分割完成:")
    logger.info(f"- 训练集: {len(train_paths)} 个样本")
    logger.info(f"- 验证集: {len(val_paths)} 个样本")
    logger.info(f"- 测试集: {len(test_paths)} 个样本")

def split_predictor_data(
    data_dir: str,
    save_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> None:
    """分割预测器数据。
    
    Args:
        data_dir: 数据目录
        save_dir: 保存目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子
    """
    # 准备数据
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    
    # 创建保存目录
    save_dir.mkdir(parents=True, exist_ok=True)
    train_dir = save_dir / "train"
    val_dir = save_dir / "val"
    test_dir = save_dir / "test"
    
    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(exist_ok=True)
        (split_dir / "states").mkdir(exist_ok=True)
    
    # 获取状态文件列表
    state_paths = list(data_dir.glob("*.json"))
    
    # 分割数据
    train_paths, temp_paths = train_test_split(
        state_paths,
        train_size=train_ratio,
        random_state=random_state
    )
    
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths = train_test_split(
        temp_paths,
        train_size=val_ratio_adjusted,
        random_state=random_state
    )
    
    # 复制数据
    for split_name, split_paths in [
        ("train", train_paths),
        ("val", val_paths),
        ("test", test_paths)
    ]:
        split_dir = save_dir / split_name
        
        for state_path in split_paths:
            # 复制状态文件
            import shutil
            shutil.copy2(
                state_path,
                split_dir / "states" / state_path.name
            )
    
    logger.info(f"预测器数据分割完成:")
    logger.info(f"- 训练集: {len(train_paths)} 个样本")
    logger.info(f"- 验证集: {len(val_paths)} 个样本")
    logger.info(f"- 测试集: {len(test_paths)} 个样本")

def main() -> None:
    """主函数。"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 分割检测器数据
    split_detector_data(
        data_dir="data/detector",
        save_dir="data/detector_split",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )
    
    # 分割预测器数据
    split_predictor_data(
        data_dir="data/predictor",
        save_dir="data/predictor_split",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )

if __name__ == "__main__":
    main()

 