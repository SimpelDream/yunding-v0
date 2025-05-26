"""YOLO-NAS训练脚本。

此脚本用于训练YOLO-NAS-S模型，用于检测棋盘上的单位、装备、状态等。
"""

import argparse
import logging
from pathlib import Path

from tftassist.detector.yolo import YOLONASDetector

logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="训练YOLO-NAS-S模型")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="数据配置文件路径"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="批次大小"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=1280,
        help="图像尺寸"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="runs/train",
        help="保存目录"
    )
    return parser.parse_args()

def main():
    """主函数。"""
    args = parse_args()
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练模型
    YOLONASDetector.train(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.img_size,
        save_dir=save_dir
    )
    
    logger.info("训练完成")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main() 