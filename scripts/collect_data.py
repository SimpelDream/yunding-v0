"""数据采集脚本。"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from tftassist.core.state import BoardState
from tftassist.vision.detector import YOLONASDetector

logger = logging.getLogger(__name__)

class DataCollector:
    """数据采集器。"""
    
    def __init__(
        self,
        save_dir: str = "data",
        interval: float = 1.0,
        max_samples: Optional[int] = None
    ) -> None:
        """初始化采集器。
        
        Args:
            save_dir: 保存目录
            interval: 采集间隔(秒)
            max_samples: 最大样本数
        """
        self.save_dir = Path(save_dir)
        self.interval = interval
        self.max_samples = max_samples
        
        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "images").mkdir(exist_ok=True)
        (self.save_dir / "states").mkdir(exist_ok=True)
        
        # 初始化检测器
        self.detector = YOLONASDetector("models/yolo.onnx")
        
        # 初始化状态
        self.state = BoardState(
            side="self",
            stage="1-1",
            rank=1,
            hp=100,
            gold=0,
            level=1,
            xp_progress=(0, 2),
            timestamp=time.time()
        )
        
        logger.info("数据采集器初始化完成")
        
    def collect(self) -> None:
        """开始采集数据。"""
        logger.info("开始采集数据")
        
        # 创建进度条
        pbar = tqdm(total=self.max_samples)
        
        try:
            while True:
                # 检查是否达到最大样本数
                if self.max_samples and pbar.n >= self.max_samples:
                    break
                    
                # 采集屏幕
                frame = self._capture_screen()
                if frame is None:
                    continue
                    
                # 检测和解析
                detections = self.detector.detect(frame)
                self.detector.parse_detections(self.state, detections)
                
                # 保存数据
                timestamp = int(time.time() * 1000)
                self._save_data(frame, timestamp)
                
                # 更新进度
                pbar.update(1)
                
                # 等待下一次采集
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            logger.info("用户中断采集")
        finally:
            pbar.close()
            logger.info("数据采集完成")
            
    def _capture_screen(self) -> Optional[np.ndarray]:
        """捕获屏幕。
        
        Returns:
            屏幕图像
        """
        try:
            # TODO: 实现屏幕捕获
            return None
        except Exception as e:
            logger.error(f"屏幕捕获失败: {e}")
            return None
            
    def _save_data(self, frame: np.ndarray, timestamp: int) -> None:
        """保存数据。
        
        Args:
            frame: 屏幕图像
            timestamp: 时间戳
        """
        # 保存图像
        image_path = self.save_dir / "images" / f"{timestamp}.jpg"
        cv2.imwrite(str(image_path), frame)
        
        # 保存状态
        state_path = self.save_dir / "states" / f"{timestamp}.json"
        # TODO: 实现状态保存

def main() -> None:
    """主函数。"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 创建采集器
    collector = DataCollector(
        save_dir="data",
        interval=1.0,
        max_samples=1000
    )
    
    # 开始采集
    collector.collect()

if __name__ == "__main__":
    main() 