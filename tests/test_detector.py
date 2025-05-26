"""检测器测试模块。"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tftassist.detector.yolo import YOLONASDetector

@pytest.fixture
def detector():
    """创建检测器实例。"""
    model_path = Path("models/yolo_nas_s.pt")
    if not model_path.exists():
        pytest.skip("模型文件不存在")
    return YOLONASDetector(model_path)

@pytest.fixture
def test_image():
    """创建测试图像。"""
    return np.random.randint(0, 255, (1280, 1280, 3), dtype=np.uint8)

def test_detector_initialization(detector):
    """测试检测器初始化。"""
    assert detector.model is not None
    assert detector.conf_threshold == 0.25
    assert detector.iou_threshold == 0.45
    assert detector.device in ["cuda", "cpu"]

def test_detection(detector, test_image):
    """测试目标检测。"""
    detections = detector.detect(test_image)
    assert isinstance(detections, list)
    
    if len(detections) > 0:
        detection = detections[0]
        assert "class" in detection
        assert "confidence" in detection
        assert "bbox" in detection
        assert isinstance(detection["class"], str)
        assert isinstance(detection["confidence"], float)
        assert isinstance(detection["bbox"], list)
        assert len(detection["bbox"]) == 4

def test_onnx_export(detector, tmp_path):
    """测试ONNX导出。"""
    save_path = tmp_path / "model.onnx"
    detector.export_onnx(save_path)
    assert save_path.exists()

def test_training(tmp_path):
    """测试模型训练。"""
    # 创建临时数据目录
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # 创建数据配置文件
    yaml_path = data_dir / "data.yaml"
    yaml_content = """
    path: ./data
    train: images/train
    val: images/val
    nc: 11
    names: ['chess_unit', 'bench_unit', 'star_1', 'star_2', 'star_3',
            'item_icon', 'status_icon', 'blue_hex', 'artifact_hex',
            'void_hex', 'portal_icon']
    """
    yaml_path.write_text(yaml_content)
    
    # 创建保存目录
    save_dir = tmp_path / "runs"
    
    # 训练模型
    YOLONASDetector.train(
        data_yaml=str(yaml_path),
        epochs=1,  # 仅训练一个epoch用于测试
        batch_size=2,
        imgsz=640,
        save_dir=save_dir
    )
    
    # 检查输出
    assert save_dir.exists()
    assert (save_dir / "yolo_nas_s_tft").exists()
    assert (save_dir / "yolo_nas_s_tft" / "weights" / "best.pt").exists() 