"""PaddleOCR 模块。

此模块实现了基于 PaddleOCR 的文本识别功能。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)


class PaddleOCREngine:
    """PaddleOCR 引擎。

    使用 PaddleOCR 进行文本识别。
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        """初始化 OCR 引擎。

        Args:
            model_dir: 模型目录路径，如果为 None 则使用默认模型
        """
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            show_log=False,
            use_gpu=True,
            det_model_dir=str(model_dir / "det") if model_dir else None,
            rec_model_dir=str(model_dir / "rec") if model_dir else None,
            cls_model_dir=str(model_dir / "cls") if model_dir else None,
        )
        logger.info("初始化 PaddleOCR 引擎")

    def recognize(self, image: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> List[Dict]:
        """识别图像中的文本。

        Args:
            image: 输入图像
            regions: 识别区域列表，每个区域为 (x1, y1, x2, y2)

        Returns:
            识别结果列表，每个结果包含文本内容和位置信息
        """
        results = []
        for region in regions:
            x1, y1, x2, y2 = region
            roi = image[y1:y2, x1:x2]
            
            try:
                ocr_result = self.ocr.ocr(roi, cls=True)
                if ocr_result and ocr_result[0]:
                    for line in ocr_result[0]:
                        text = line[1][0]
                        confidence = line[1][1]
                        box = line[0]
                        
                        # 将相对坐标转换为绝对坐标
                        abs_box = [
                            (x1 + point[0], y1 + point[1])
                            for point in box
                        ]
                        
                        results.append({
                            "text": text,
                            "confidence": confidence,
                            "box": abs_box
                        })
            except Exception as e:
                logger.error(f"OCR 识别失败: {e}")
                
        return results

    def recognize_single(self, image: np.ndarray) -> List[Dict]:
        """识别整个图像中的文本。

        Args:
            image: 输入图像

        Returns:
            识别结果列表
        """
        try:
            ocr_result = self.ocr.ocr(image, cls=True)
            if not ocr_result or not ocr_result[0]:
                return []
                
            results = []
            for line in ocr_result[0]:
                text = line[1][0]
                confidence = line[1][1]
                box = line[0]
                
                results.append({
                    "text": text,
                    "confidence": confidence,
                    "box": box
                })
                
            return results
        except Exception as e:
            logger.error(f"OCR 识别失败: {e}")
            return [] 