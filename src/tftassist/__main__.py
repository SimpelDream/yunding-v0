"""程序入口点。"""

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .capture.screen import ScreenCapturer
from .core.feature import extract_features
from .core.state import BoardState
from .detector.yolo import YOLODetector
from .ocr.paddle import PaddleOCR
from .predictor.lgbm import LGBMPredictor
from .ui.overlay import OverlayWindow

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main() -> None:
    """主函数。"""
    try:
        # 初始化模型
        model_dir = Path('models')
        detector = YOLODetector(str(model_dir / 'yolo.onnx'))
        ocr = PaddleOCR()
        predictor = LGBMPredictor(str(model_dir / 'lgbm.txt'))
        
        # 初始化屏幕捕获
        capture = ScreenCapturer()
        
        # 初始化UI
        overlay = OverlayWindow()
        overlay.show()
        
        # 主循环
        while True:
            try:
                # 捕获屏幕
                frame = await capture.capture()
                if frame is None:
                    continue
                
                # 检测和OCR
                detector.update_state(overlay.central_widget.state, frame)
                await ocr.update_state(overlay.central_widget.state, frame)
                
                # 提取特征并预测
                features = extract_features(overlay.central_widget.state)
                pred, importance = predictor.predict(features)
                
                # 更新UI
                overlay.central_widget.update_state(overlay.central_widget.state)
                
                # 处理事件
                cv2.waitKey(1)
                
            except Exception as e:
                logger.error(f"主循环出错: {e}")
                
    except Exception as e:
        logger.error(f"程序出错: {e}")
        sys.exit(1)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main()) 