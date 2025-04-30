"""TFT 辅助工具主程序入口。

此模块实现了程序的主要流程。
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from tftassist.capture.screen import ScreenCapturer
from tftassist.core.state import BoardState
from tftassist.detector.yolo import YOLODetector
from tftassist.ocr.paddle import PaddleOCREngine
from tftassist.predictor.lgbm import LGBMPredictor
from tftassist.ui.overlay import OverlayWindow

logger = logging.getLogger(__name__)


async def main() -> None:
    """主程序入口。"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="TFT 云顶之弈辅助工具")
    parser.add_argument("--demo", action="store_true", help="演示模式")
    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # 初始化 Qt 应用
        app = QApplication(sys.argv)

        # 初始化组件
        capturer = ScreenCapturer()
        detector = YOLODetector(Path("models/yolo.pt"))
        ocr = PaddleOCREngine(Path("models/ocr"))
        predictor = LGBMPredictor(Path("models/lgbm.txt"))
        overlay = OverlayWindow()
        overlay.show()

        # 主循环
        while True:
            # 捕获屏幕
            frame = await capturer.capture()

            # 检测目标
            detections = detector.detect(frame)

            # OCR 识别
            ocr_results = ocr.recognize(frame, [d["bbox"] for d in detections])

            # 解析游戏状态
            state = BoardState(
                side="self",
                stage="1-1",
                phase_timer=None,
                rank=1,
                hp=100,
                gold=0,
                level=1,
                xp_progress=(0, 2),
                shop_odds={},
                shop_cards=[],
                traits={},
                inactive_traits={},
                units=[],
                bench_units=[],
                augments=[],
                hex_map={},
                combo_meter=None,
                enemy_hp_vec=[],
                fps=None,
                ping_ms=None,
                timestamp=0.0,
            )

            # 预测胜率和伤害
            win_rate, damage = predictor.predict(state)

            # 更新 UI
            overlay.update_state(win_rate, damage, state)

            # 处理 Qt 事件
            app.processEvents()

            # 演示模式延迟
            if args.demo:
                await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("程序退出")
    finally:
        capturer.close()
        app.quit()


if __name__ == "__main__":
    asyncio.run(main()) 