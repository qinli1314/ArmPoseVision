"""
ArmPoseVision - 手臂姿态视觉识别系统
基于单目摄像头的人体手臂姿态实时识别与角度分析

启动方式:
    # USB 摄像头模式（默认）
    python main.py

    # 视频文件回放模式（前置自拍，自动镜像）
    python main.py --video path/to/video.mp4

    # 视频文件回放模式（后置拍摄，不镜像）
    python main.py --video path/to/video.mp4 --rear

    # 视频文件循环回放
    python main.py --video path/to/video.mp4 --loop
"""

import sys
import os
import argparse
import logging

# 确保项目根目录在路径中
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ── 日志配置 ──
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, "app.log"),
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("main")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="ArmPoseVision - 手臂姿态视觉识别系统"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="视频文件路径，代替 USB 摄像头输入"
    )
    parser.add_argument(
        "--rear", "-r",
        action="store_true",
        help="后置摄像头拍摄（不镜像翻转），默认前置（自动镜像）",
    )
    parser.add_argument("--loop", "-l", action="store_true", help="视频文件循环播放")
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="指定摄像头设备 ID（默认使用配置文件中的值）",
    )
    return parser.parse_args()


def main():
    """程序入口"""
    from PyQt5.QtWidgets import QApplication
    from gui.main_window import MainWindow

    args = parse_args()

    # 高 DPI 支持
    QApplication.setAttribute(0x10001)  # Qt.AA_EnableHighDpiScaling
    QApplication.setAttribute(0x10002)  # Qt.AA_UseHighDpiPixmaps

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 暗色主题
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #1e1e1e;
            color: #e0e0e0;
        }
        QComboBox {
            background-color: #2d2d2d; color: #e0e0e0;
            padding: 4px; border: 1px solid #444;
            border-radius: 3px;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox QAbstractItemView {
            background-color: #2d2d2d; color: #e0e0e0;
            selection-background-color: #3a3a3a;
        }
        QSplitter::handle {
            background-color: #333; width: 2px;
        }
        QStatusBar {
            background-color: #252525;
            border-top: 1px solid #333;
        }
        QScrollBar:vertical {
            background: #1a1a1a; width: 8px;
        }
        QScrollBar::handle:vertical {
            background: #444; border-radius: 4px;
        }
    """)

    # 摄像头类型 → 镜像设置
    # 前置（自拍）：手机自动镜像了，需要翻回来 mirror=True
    # 后置（别人拍）：画面是自然的，不需要翻 mirror=False
    camera_type = "rear" if args.rear else "front"

    # 将命令行参数转为字典传给 MainWindow
    cli_opts = {
        "video_path": args.video,
        "loop_video": args.loop,
        "mirror": not args.rear,  # 后置→不镜像，前置→镜像
        "camera_type": camera_type,
        "camera_id": args.camera,
    }

    window = MainWindow(cli_opts=cli_opts)
    window.show()

    logger.info("ArmPoseVision 启动")
    logger.info(f"模式: {'视频回放' if args.video else 'USB 摄像头'}"
                f" | {'后置' if args.rear else '前置'} "
                f"| {'镜像关' if args.rear else '镜像开'}")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
