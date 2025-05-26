"""目标检测模块。"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

from ..core.state import BoardState, Unit

logger = logging.getLogger(__name__)

# 检测类别
DETECTION_CLASSES = {
    # 单位
    0: "Ahri",
    1: "Akali",
    2: "Annie",
    3: "Ashe",
    4: "Blitzcrank",
    5: "Brand",
    6: "Braum",
    7: "Caitlyn",
    8: "Camille",
    9: "Cassiopeia",
    # 装备
    10: "B.F. Sword",
    11: "Chain Vest",
    12: "Giant's Belt",
    13: "Needlessly Large Rod",
    14: "Negatron Cloak",
    15: "Recurve Bow",
    16: "Spatula",
    17: "Tear of the Goddess",
    # 六边形
    18: "Mage Hex",
    19: "Assassin Hex",
    20: "Tank Hex",
    21: "Support Hex",
    22: "Marksman Hex",
    23: "Fighter Hex"
}

class YOLONASDetector:
    """YOLO-NAS检测器。"""
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ) -> None:
        """初始化检测器。
        
        Args:
            model_path: 模型路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            device: 设备
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"YOLO-NAS检测器初始化完成, 设备: {self.device}")
        
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """执行检测。
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果列表
        """
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    "class": DETECTION_CLASSES[cls],
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })
                
        return detections
        
    def parse_detections(
        self,
        state: BoardState,
        detections: List[Dict[str, Any]]
    ) -> None:
        """解析检测结果。
        
        Args:
            state: 游戏状态
            detections: 检测结果列表
        """
        # 清空现有状态
        state.units.clear()
        state.bench_units.clear()
        state.hex_map.clear()
        
        # 解析检测结果
        for det in detections:
            cls = det["class"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["bbox"]
            
            # 计算中心点
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 根据类别处理
            if cls in ["Ahri", "Akali", "Annie", "Ashe", "Blitzcrank", "Brand", "Braum", "Caitlyn", "Camille", "Cassiopeia"]:
                # 单位
                unit = Unit(
                    name=cls,
                    star=1,  # 默认1星
                    items=[],  # 默认无装备
                    position=(int(center_x), int(center_y)),
                    hp_pct=1.0,  # 默认满血
                    shield_pct=0.0,  # 默认无护盾
                    status_tags=[]  # 默认无状态
                )
                
                # 根据位置判断是场上还是备战区
                if center_y < 400:  # 假设400是分界线
                    state.units.append(unit)
                else:
                    state.bench_units.append(unit)
                    
            elif cls.endswith("Hex"):
                # 六边形
                hex_type = cls[:-4]  # 去掉"Hex"后缀
                state.hex_map[(int(center_x), int(center_y))] = hex_type
                
    def export_onnx(self, save_path: str) -> None:
        """导出ONNX模型。
        
        Args:
            save_path: 保存路径
        """
        self.model.export(format="onnx", save_path=save_path)
        logger.info(f"已导出ONNX模型: {save_path}")
        
    @staticmethod
    def train(
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        device: Optional[str] = None
    ) -> None:
        """训练模型。
        
        Args:
            data_yaml: 数据配置文件路径
            epochs: 训练轮数
            batch_size: 批次大小
            img_size: 图像大小
            device: 设备
        """
        model = YOLO("yolo_nas_s")
        model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device
        )
        logger.info("模型训练完成") 