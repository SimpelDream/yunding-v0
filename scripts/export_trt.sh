#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 导出ONNX模型
python -c "
from tftassist.detector.yolo import YOLONASDetector
detector = YOLONASDetector('models/yolo_nas_s.pt')
detector.export_onnx('models/yolo_nas_s.onnx')
"

# 使用trtexec转换为TensorRT引擎
trtexec \
    --onnx=models/yolo_nas_s.onnx \
    --saveEngine=models/yolo_nas_s.engine \
    --fp16 \
    --workspace=4096 \
    --verbose \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:1x3x1280x1280 \
    --maxShapes=images:1x3x1280x1280 