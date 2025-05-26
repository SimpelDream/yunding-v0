"""YOLO-NAS检测器模块。

此模块实现了基于YOLO-NAS-S的目标检测功能，用于识别棋盘上的单位、装备、状态等。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO
import cv2
from ..core.state import BoardState, Unit

logger = logging.getLogger(__name__)

# 检测类别定义
DETECTION_CLASSES = [
    "chess_unit",  # 棋盘单位
    "bench_unit",  # 备战区单位
    "star_1",      # 1星
    "star_2",      # 2星
    "star_3",      # 3星
    "item_icon",   # 装备图标
    "status_icon", # 状态图标
    "blue_hex",    # 蓝色海克斯
    "artifact_hex",# 神器海克斯
    "void_hex",    # 虚空海克斯
    "portal_icon"  # 传送门图标
]

class YOLONASDetector:
    """YOLO-NAS-S检测器类。
    
    Attributes:
        model: YOLO-NAS-S模型
        conf_threshold: 置信度阈值
        iou_threshold: IOU阈值
        device: 运行设备
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """初始化检测器。
        
        Args:
            model_path: 模型路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            device: 运行设备
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # 设置模型参数
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        self.model.to(device)
        
        logger.info(f"YOLO-NAS-S检测器初始化完成，使用设备: {device}")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """执行目标检测。
        
        Args:
            image: 输入图像，BGR格式
            
        Returns:
            检测结果列表，每个结果包含类别、置信度、边界框等信息
        """
        results = self.model(image, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy.tolist()[0]
            
            detections.append({
                "class": DETECTION_CLASSES[cls],
                "confidence": conf,
                "bbox": xyxy
            })
        
        return detections
    
    def export_onnx(self, save_path: Union[str, Path]) -> None:
        """导出ONNX模型。
        
        Args:
            save_path: 保存路径
        """
        self.model.export(format="onnx", imgsz=1280)
        logger.info(f"模型已导出到: {save_path}")
    
    @staticmethod
    def train(
        data_yaml: Union[str, Path],
        epochs: int = 60,
        batch_size: int = 16,
        imgsz: int = 1280,
        save_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """训练YOLO-NAS-S模型。
        
        Args:
            data_yaml: 数据配置文件路径
            epochs: 训练轮数
            batch_size: 批次大小
            imgsz: 图像尺寸
            save_dir: 保存目录
        """
        model = YOLO("yolo_nas_s.pt")
        
        # 训练参数
        train_args = {
            "data": data_yaml,
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": imgsz,
            "device": "0" if torch.cuda.is_available() else "cpu",
            "optimizer": "AdamW",
            "amp": True,  # 启用混合精度训练
            "project": save_dir if save_dir else "runs/train",
            "name": "yolo_nas_s_tft"
        }
        
        # 开始训练
        model.train(**train_args)
        logger.info("模型训练完成")

class YOLODetector:
    """YOLO 目标检测器类。"""

    def __init__(self, model_path: str) -> None:
        """初始化检测器。

        Args:
            model_path: 模型路径
        """
        self.model = YOLO(model_path)
        logger.info(f"加载YOLO模型: {model_path}")

    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float, int]]:
        """检测目标。
        
        Args:
            frame: 输入图像
            
        Returns:
            检测结果列表，每个元素为 (x1, y1, x2, y2, conf, cls)
        """
        # 预处理
        input_size = (640, 640)
        img = cv2.resize(frame.copy(), input_size).astype(np.float32)
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.ascontiguousarray(img)
        img = img / 255.0
        
        # 推理
        results = self.model(img)[0]
        boxes = results.boxes
        
        # 后处理
        detections: List[Tuple[float, float, float, float, float, int]] = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append((x1, y1, x2, y2, conf, cls))
            
        return detections

    def update_state(self, state: BoardState, frame: np.ndarray) -> None:
        """更新游戏状态。
        
        Args:
            state: 游戏状态
            frame: 输入图像
        """
        # 检测目标
        detections = self.detect(frame.copy())
        
        # 更新状态
        state.board_height, state.board_width = frame.shape[:2]
        
        # 处理检测结果
        for x1, y1, x2, y2, conf, cls in detections:
            if conf < 0.5:
                continue
                
            # 创建单位
            unit = Unit(
                name="",  # 需要OCR识别
                traits=[],
                position=(int(x1), int(y1)),
                size=(int(x2-x1), int(y2-y1))
            )
            
            # 更新状态
            if cls == 0:  # 棋盘单位
                state.units.append(unit)
            elif cls == 1:  # 备战区单位
                state.bench_units.append(unit)

    def export_onnx(self, output_path: Path) -> None:
        """导出模型为 ONNX 格式。

        Args:
            output_path: 输出文件路径
        """
        self.model.export(format="onnx", imgsz=1280, half=True)
        logger.info(f"导出 ONNX 模型: {output_path}")


class ONNXDetector:
    """ONNX 运行时检测器。

    使用 ONNX 运行时进行目标检测。
    """

    def __init__(self, model_path: Path) -> None:
        """初始化检测器。

        Args:
            model_path: ONNX 模型文件路径
        """
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info(f"加载 ONNX 模型: {model_path}")

    def detect(self, image: np.ndarray) -> List[Dict]:
        """检测图像中的目标。

        Args:
            image: 输入图像

        Returns:
            检测结果列表，每个结果包含类别、置信度和边界框
        """
        # 预处理图像
        input_tensor = self._preprocess(image)
        
        # 运行推理
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # 后处理结果
        detections = self._postprocess(outputs)
        return detections

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理输入图像。

        Args:
            image: 输入图像

        Returns:
            预处理后的张量
        """
        # TODO: 实现图像预处理
        raise NotImplementedError

    def _postprocess(self, outputs: List[np.ndarray]) -> List[Dict]:
        """后处理模型输出。

        Args:
            outputs: 模型输出

        Returns:
            检测结果列表
        """
        # TODO: 实现结果后处理
        raise NotImplementedError 