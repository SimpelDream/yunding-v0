"""目标检测模型训练脚本。

此脚本用于训练 YOLO 目标检测模型。
"""

import logging
from pathlib import Path

import yaml
from ultralytics import YOLO

logger = logging.getLogger(__name__)

def train(config_path: str) -> None:
    """训练模型。

    Args:
        config_path: 配置文件路径
    """
    try:
        # 加载配置
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # 创建输出目录
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型
        model = YOLO(config["model"])
        
        # 训练模型
        model.train(
            data=config["data"],
            epochs=config["epochs"],
            batch=config["batch_size"],
            imgsz=config["image_size"],
            device=config["device"],
            workers=config["workers"],
            project=str(output_dir),
            name=config["name"],
            exist_ok=config["exist_ok"],
            pretrained=config["pretrained"],
            verbose=config["verbose"],
            seed=config["seed"]
        )
        
        # 导出模型
        model.export(
            format="onnx",
            imgsz=config["image_size"],
            dynamic=True,
            simplify=True
        )
        
        logger.info("模型训练完成")
        
    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        raise

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 训练模型
    train("configs/train_det.yaml") 