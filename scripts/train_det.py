"""YOLO-NAS 训练脚本。

此脚本用于训练目标检测模型。
"""

import logging
from pathlib import Path

import yaml
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def main() -> None:
    """主函数。"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 加载配置
    config_path = Path("configs/det.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 初始化模型
    model = YOLO("yolo_nas_s.pt")

    # 训练模型
    model.train(
        data=config["data"],
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        optimizer=config["optimizer"],
        lr0=config["lr0"],
        lrf=config["lrf"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        warmup_epochs=config["warmup_epochs"],
        warmup_momentum=config["warmup_momentum"],
        warmup_bias_lr=config["warmup_bias_lr"],
        box=config["box"],
        cls=config["cls"],
        dfl=config["dfl"],
        fl_gamma=config["fl_gamma"],
        label_smoothing=config["label_smoothing"],
        nbs=config["nbs"],
        overlap_mask=config["overlap_mask"],
        mask_ratio=config["mask_ratio"],
        dropout=config["dropout"],
        val=config["val"],
        save=config["save"],
        save_period=config["save_period"],
        cache=config["cache"],
        device=config["device"],
        workers=config["workers"],
        project=config["project"],
        name=config["name"],
        exist_ok=config["exist_ok"],
        pretrained=config["pretrained"],
        optimizer=config["optimizer"],
        verbose=config["verbose"],
        seed=config["seed"],
        deterministic=config["deterministic"],
        single_cls=config["single_cls"],
        rect=config["rect"],
        cos_lr=config["cos_lr"],
        close_mosaic=config["close_mosaic"],
        resume=config["resume"],
        amp=config["amp"],
        fraction=config["fraction"],
        freeze=config["freeze"],
        patience=config["patience"],
        plots=config["plots"],
    )

    # 导出模型
    model.export(format="onnx", imgsz=1280, half=True)


if __name__ == "__main__":
    main() 