"""
GUI - 主窗口 (Main Window)
职责：整体界面布局，集成视频、角度面板、日志、控制按钮
纯视觉版本 - 无需机械臂硬件
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Optional
from threading import Thread, Event

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QComboBox, QTextEdit, QStatusBar,
    QLabel, QSplitter, QMessageBox, QApplication,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon

from vision.camera import CameraCapture
from vision.pose_detector import PoseDetector
from vision.angle_extractor import AngleExtractor
from filter.motion_filter import MotionFilter
from mapping.mapper import Mapper
from gui.video_widget import VideoWidget
from gui.angle_panel import AnglePanel

logger = logging.getLogger(__name__)

# 配置文件路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_CONFIG_PATH = os.path.join(BASE_DIR, "config", "app_config.json")
LOG_DIR = os.path.join(BASE_DIR, "logs")


class MainWindow(QMainWindow):
    """主窗口 - 纯视觉版"""

    # 信号：跨线程更新 GUI
    frame_ready = pyqtSignal(object, object, float)   # (原始帧, 骨架帧, fps)
    angles_ready = pyqtSignal(object, object, object, bool)  # (角度, hold_mask, 电机指令, detected)
    log_signal = pyqtSignal(str)                       # 日志消息

    def __init__(self, cli_opts: Optional[dict] = None):
        super().__init__()
        self.setWindowTitle("手臂姿态视觉识别系统 - ArmPoseVision v1.0")
        self.setMinimumSize(1200, 750)

        # ── 命令行选项 ──
        self.cli_opts = cli_opts or {}

        # ── 加载配置 ──
        self.config = self._load_config()

        # ── 成员变量 ──
        self.camera: Optional[CameraCapture] = None
        self.pose_detector: Optional[PoseDetector] = None
        self.angle_extractor: Optional[AngleExtractor] = None
        self.motion_filter: Optional[MotionFilter] = None
        self.mapper: Optional[Mapper] = None

        self._running = False
        self._capture_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._is_video_mode = self.cli_opts.get("video_path") is not None
        self._config_mtime = 0  # 配置文件修改时间

        # 显示镜像控制：推理用原图，显示才镜像
        # 前置摄像头：显示镜像（用户看着自然），推理用原图
        # 后置摄像头：不镜像
        cam_type = self.cli_opts.get("camera_type", "front")
        self._mirror_display = (cam_type == "front")

        # ── 初始化 UI ──
        self._init_ui()
        self._init_modules()

        # ── GUI 定时器 ──
        self._fps_timer = QTimer()
        self._fps_timer.timeout.connect(self._update_status_fps)
        self._fps_timer.start(1000)

        # ── 配置文件热重载定时器 ──
        self._config_watch_timer = QTimer()
        self._config_watch_timer.timeout.connect(self._check_config_hot_reload)
        self._config_watch_timer.start(3000)  # 每3秒检查一次

        # ── 视频文件模式自动启动 ──
        if self._is_video_mode:
            self._start_video_mode()

        logger.info("主窗口初始化完成")

    # ══════════════════════════════════════
    #  UI 初始化
    # ══════════════════════════════════════

    def _init_ui(self):
        """初始化界面布局"""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # ── 顶部工具栏 ──
        toolbar = QHBoxLayout()

        self.cam_combo = QComboBox()
        self.cam_combo.setMinimumWidth(280)
        self.cam_combo.setPlaceholderText("选择摄像头...")

        self.btn_start = QPushButton("▶ 开始采集")
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #2d7d46; color: white;
                padding: 6px 24px; border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #3a9d5a; }
            QPushButton:disabled { background-color: #555; }
        """)

        self.btn_stop = QPushButton("■ 停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #c0392b; color: white;
                padding: 6px 24px; border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #e74c3c; }
            QPushButton:disabled { background-color: #555; }
        """)

        self.btn_reload = QPushButton("⟳ 重载映射")
        self.btn_reload.setStyleSheet("""
            QPushButton {
                background-color: #2c3e50; color: white;
                padding: 6px 16px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #34495e; }
        """)

        # 模式标签（初始显示）
        self.mode_label = QLabel("⚡ 纯视觉模式")
        self.mode_label.setStyleSheet(
            "color: #00cc88; font-weight: bold; padding: 4px 12px;"
            "border: 1px solid #00cc88; border-radius: 4px;"
        )

        # 摄像头类型选择
        self.cam_type_combo = QComboBox()
        self.cam_type_combo.setMinimumWidth(100)
        self.cam_type_combo.addItem("📱 前置 (镜像)", "front")
        self.cam_type_combo.addItem("📱 后置 (原画)", "rear")
        self.cam_type_combo.setStyleSheet("""
            QComboBox {
                background-color: #2d2d2d; color: #88ccff;
                padding: 4px; border: 1px solid #88ccff;
                border-radius: 4px; font-weight: bold;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d; color: #e0e0e0;
                selection-background-color: #3a3a3a;
            }
        """)
        # 根据命令行参数设置默认选择
        default_type = self.cli_opts.get("camera_type", "front")
        idx = self.cam_type_combo.findData(default_type)
        if idx >= 0:
            self.cam_type_combo.setCurrentIndex(idx)
        self.cam_type_combo.currentIndexChanged.connect(self._on_camera_type_changed)

        # 左右臂切换
        self.arm_combo = QComboBox()
        self.arm_combo.setMinimumWidth(80)
        self.arm_combo.addItem("🦾 右臂", "right")
        self.arm_combo.addItem("🦾 左臂", "left")
        default_arm = self.config.get("pose", {}).get("arm", "right")
        idx = self.arm_combo.findData(default_arm)
        if idx >= 0:
            self.arm_combo.setCurrentIndex(idx)
        self.arm_combo.setStyleSheet("""
            QComboBox {
                background-color: #2d2d2d; color: #ffcc88;
                padding: 4px; border: 1px solid #ffcc88;
                border-radius: 4px; font-weight: bold;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d; color: #e0e0e0;
                selection-background-color: #3a3a3a;
            }
        """)
        self.arm_combo.currentIndexChanged.connect(self._on_arm_changed)

        if self._is_video_mode:
            # 视频文件模式：简化工具栏
            video_name = os.path.basename(self.cli_opts["video_path"])
            toolbar.addWidget(QLabel("🎬"))
            self.source_label = QLabel(f"📁 {video_name}")
            self.source_label.setStyleSheet("color: #ffcc00; font-weight: bold;")
            toolbar.addWidget(self.source_label)
            toolbar.addSpacing(8)
            toolbar.addWidget(self.btn_start)
            toolbar.addWidget(self.btn_stop)
            toolbar.addSpacing(6)
            toolbar.addWidget(self.arm_combo)
            toolbar.addSpacing(6)
            toolbar.addWidget(self.cam_type_combo)
            toolbar.addSpacing(6)
            toolbar.addStretch()
            self.mode_label.setText("🎬 视频回放模式")
            self.mode_label.setStyleSheet(
                "color: #ffcc00; font-weight: bold; padding: 4px 12px;"
                "border: 1px solid #ffcc00; border-radius: 4px;"
            )
            self.cam_combo.hide()
        else:
            # 摄像头模式
            toolbar.addWidget(QLabel("📷"))
            toolbar.addWidget(self.cam_combo)
            toolbar.addSpacing(8)
            toolbar.addWidget(self.btn_start)
            toolbar.addWidget(self.btn_stop)
            toolbar.addSpacing(6)
            toolbar.addWidget(self.arm_combo)
            toolbar.addSpacing(6)
            toolbar.addWidget(self.cam_type_combo)
            toolbar.addSpacing(6)
            toolbar.addWidget(self.btn_reload)
            toolbar.addStretch()

        toolbar.addWidget(self.mode_label)

        main_layout.addLayout(toolbar)

        # ── 主内容区：视频（左）+ 角度面板（右） ──
        content = QSplitter(Qt.Horizontal)

        # 左侧：视频
        self.video_widget = VideoWidget()
        content.addWidget(self.video_widget)

        # 右侧：角度面板
        self.angle_panel = AnglePanel()
        content.addWidget(self.angle_panel)

        # 设置分割比例
        content.setSizes([800, 300])
        content.setStretchFactor(0, 3)
        content.setStretchFactor(1, 1)

        main_layout.addWidget(content, stretch=1)

        # ── 底部日志 ──
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(180)
        self.log_output.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #c0c0c0;
                font-family: Consolas, monospace;
                font-size: 13px;
                border: 1px solid #333;
            }
        """)
        main_layout.addWidget(self.log_output)

        # ── 状态栏 ──
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("font-weight: bold; padding: 0 12px;")
        self.status_bar.addPermanentWidget(self.fps_label)

        # ── 信号绑定 ──
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_reload.clicked.connect(self._on_reload_mapping)

        # 日志信号连接
        self.log_signal.connect(self._append_log)
        self.frame_ready.connect(self._on_frame_ready)
        self.angles_ready.connect(self._on_angles_ready)

        # 初始日志
        cam_type = self.cli_opts.get("camera_type", "front")
        mirror_label = "镜像开" if self.cli_opts.get("mirror", True) else "镜像关"
        if self._is_video_mode:
            video_name = os.path.basename(self.cli_opts["video_path"])
            self.log(f"系统就绪 — 视频回放 [{video_name}] | {cam_type} | {mirror_label}")
        else:
            self.log(f"系统就绪 — 纯视觉模式 | {cam_type} | {mirror_label}")

    # ══════════════════════════════════════
    #  模块初始化
    # ══════════════════════════════════════

    def _init_modules(self):
        """初始化各功能模块"""
        # 角度映射（仅用于角度限位显示和映射演示）
        self.mapper = Mapper()

        # 姿态检测 & 角度提取
        self.pose_detector = PoseDetector(
            detection_confidence=self.config.get("pose", {}).get("detection_confidence", 0.5),
            tracking_confidence=self.config.get("pose", {}).get("tracking_confidence", 0.5),
            arm=self.config.get("pose", {}).get("arm", "right"),
        )
        self.angle_extractor = AngleExtractor(
            arm=self.config.get("pose", {}).get("arm", "right"),
        )

        # 滤波
        filter_cfg = self.config.get("filter", {})
        self.motion_filter = MotionFilter(
            dead_zone_deg=filter_cfg.get("dead_zone_deg", 5.0),
            ema_alpha=filter_cfg.get("ema_alpha", 0.3),
            hold_timeout_ms=filter_cfg.get("hold_timeout_ms", 500),
        )

        # 摄像头模式才扫描设备
        if not self._is_video_mode:
            self._refresh_cameras()

    # ══════════════════════════════════════
    #  配置管理
    # ══════════════════════════════════════

    def _check_config_hot_reload(self):
        """检查 app_config.json 是否变更，有则热重载"""
        if not os.path.exists(APP_CONFIG_PATH):
            return
        try:
            new_mtime = os.path.getmtime(APP_CONFIG_PATH)
            if new_mtime > self._config_mtime:
                if self._config_mtime > 0:
                    self.log("⟳ 应用配置已变更，执行热重载")
                self._config_mtime = new_mtime
                new_config = self._load_config()
                if new_config:
                    self.config = new_config
                    logger.info("应用配置已热重载")
        except Exception as e:
            logger.warning(f"配置热重载检查异常: {e}")

    def _load_config(self) -> dict:
        """加载应用配置，记录文件修改时间"""
        # 记录修改时间
        if os.path.exists(APP_CONFIG_PATH):
            self._config_mtime = os.path.getmtime(APP_CONFIG_PATH)
        default = {
            "camera": {
                "device_id": 0,
                "resolution": [1280, 720],
                "target_fps": 30,
                "mirror": True
            },
            "filter": {
                "dead_zone_deg": 5.0,
                "ema_alpha": 0.3,
                "hold_timeout_ms": 500
            },
            "pose": {
                "arm": "right",
                "detection_confidence": 0.5,
                "tracking_confidence": 0.5
            },
            "logging": {
                "angle_log_enabled": True,
                "log_dir": "logs"
            },
        }
        if not os.path.exists(APP_CONFIG_PATH):
            return default
        try:
            with open(APP_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # 合并（保留默认值）
            for key in default:
                if key not in cfg:
                    cfg[key] = default[key]
                elif isinstance(default[key], dict):
                    for sub in default[key]:
                        if sub not in cfg[key]:
                            cfg[key][sub] = default[key][sub]
            return cfg
        except Exception as e:
            logger.warning(f"配置文件加载失败: {e}，使用默认配置")
            return default

    def _refresh_cameras(self):
        """刷新摄像头列表"""
        self.cam_combo.clear()
        cams = CameraCapture.list_cameras_with_name()
        if not cams:
            self.cam_combo.addItem("未检测到摄像头", -1)
            self.btn_start.setEnabled(False)
        else:
            for cam in cams:
                label = f"Camera {cam['id']} - {cam['resolution'][0]}x{cam['resolution'][1]}"
                self.cam_combo.addItem(label, cam["id"])
            self.btn_start.setEnabled(True)

    # ══════════════════════════════════════
    #  采集控制
    # ══════════════════════════════════════

    def _on_start(self):
        """开始采集/回放"""
        if self._is_video_mode:
            # 视频文件模式：直接读取已配置的视频
            video_path = self.cli_opts["video_path"]
            try:
                self.camera = CameraCapture.from_video(
                    video_path=video_path,
                    mirror=False,  # 镜像由 GUI 控制，CameraCapture 只提供原图
                    loop=self.cli_opts.get("loop_video", True),
                )
                self.camera.start()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法打开视频文件: {e}")
                return
        else:
            # USB 摄像头模式
            if self.cam_combo.currentData() is None or self.cam_combo.currentData() < 0:
                QMessageBox.warning(self, "提示", "请先选择有效的摄像头")
                return

            device_id = self.cam_combo.currentData()
            cam_cfg = self.config.get("camera", {})

            try:
                self.camera = CameraCapture(
                    device_id=device_id,
                    resolution=tuple(cam_cfg.get("resolution", [1280, 720])),
                    target_fps=cam_cfg.get("target_fps", 30),
                    mirror=False,  # 镜像由 GUI 控制，CameraCapture 只提供原图
                )
                self.camera.start()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法打开摄像头: {e}")
                return

        self._running = True
        self._stop_event.clear()

        # 启动处理线程
        self._capture_thread = Thread(target=self._process_loop, name="ProcessLoop", daemon=True)
        self._capture_thread.start()

        # 更新 UI
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        if self._is_video_mode:
            self.log(f"视频回放已开始: {os.path.basename(self.cli_opts['video_path'])}")
        else:
            self.cam_combo.setEnabled(False)
            self.log("采集已开始")

    def _on_stop(self):
        """停止采集/回放"""
        self._running = False
        self._stop_event.set()

        if self.camera:
            self.camera.stop()
            self.camera = None

        # 重置滤波
        if self.motion_filter:
            self.motion_filter.reset()

        # 更新 UI
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if not self._is_video_mode:
            self.cam_combo.setEnabled(True)
        self.video_widget.update_frame(None)
        self.log("已停止")

    def _on_reload_mapping(self):
        """重载映射配置"""
        if self.mapper:
            self.mapper.load_config()
            self.angle_panel.reload_limits()
            self.log("⟳ 映射配置及限位已重载")

    def _on_camera_type_changed(self, index: int):
        """切换前置/后置摄像头模式"""
        cam_type = self.cam_type_combo.itemData(index)
        self._mirror_display = (cam_type == "front")
        mode_str = "前置 (显示镜像)" if self._mirror_display else "后置 (原画)"
        self.log(f"📷 切换摄像头模式: {mode_str}")
        logger.info(f"摄像头模式切换: {mode_str}, mirror_display={self._mirror_display}")

    def _on_arm_changed(self, index: int):
        """切换追踪手臂"""
        arm = self.arm_combo.itemData(index)
        if self.pose_detector:
            self.pose_detector.switch_arm(arm)
        if self.angle_extractor:
            self.angle_extractor.set_arm(arm)
        self.angle_panel.set_arm(arm)
        # 重置滤波器状态，避免新旧手臂数据混用
        if self.motion_filter:
            self.motion_filter.reset()
        arm_str = "右臂" if arm == "right" else "左臂"
        self.log(f"🦾 切换追踪手臂: {arm_str}")
        logger.info(f"切换追踪手臂: {arm_str}")

    # ══════════════════════════════════════
    #  视频文件模式
    # ══════════════════════════════════════

    def _start_video_mode(self):
        """视频文件模式：打开文件并自动启动回放"""
        video_path = self.cli_opts["video_path"]
        if not os.path.exists(video_path):
            self.log(f"❌ 视频文件不存在: {video_path}")
            logger.error(f"视频文件不存在: {video_path}")
            return

        self.log(f"🎬 加载视频文件: {os.path.basename(video_path)}")
        QTimer.singleShot(500, self._on_start)  # 延迟半秒自动启动

    # ══════════════════════════════════════
    #  处理循环（独立线程）
    # ══════════════════════════════════════

    def _process_loop(self):
        """采集→推理→滤波→映射→显示 主循环"""
        self.log("处理线程已启动")

        while self._running and not self._stop_event.is_set():
            if self.camera is None:
                break

            # 1. 获取原始帧（未镜像）
            raw_frame = self.camera.read_frame()
            if raw_frame is None:
                continue

            fps = self.camera.fps
            h, w = raw_frame.shape[:2]

            # 2. 构建显示帧（前置需要镜像，后置不需要）
            if self._mirror_display:
                display_frame = cv2.flip(raw_frame, 1)
            else:
                display_frame = raw_frame.copy()

            # 3. 姿态检测 — 始终在原始帧上推理，确保左右手正确
            try:
                # 前置摄像头画面是镜像的 → 手部左右标签会互换
                result = self.pose_detector.detect(raw_frame, mirror_hand=self._mirror_display)
            except Exception as e:
                logger.error(f"姿态检测异常: {e}")
                result = None

            # 4. 绘制骨架
            if result:
                # 如果显示画面是镜像的，骨架关键点也要镜像
                if self._mirror_display:
                    display_keypoints = self.pose_detector.mirror_keypoints(
                        result["keypoints"], w
                    )
                    result["keypoints"] = display_keypoints
                skeleton_frame = self.pose_detector.draw_skeleton(display_frame, result)
            else:
                skeleton_frame = display_frame.copy()
                cv2.putText(skeleton_frame, "未检测到人体", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 5. 发送到 GUI 显示（在叠加角度之前先发原始骨架帧）
            self.frame_ready.emit(raw_frame, skeleton_frame, fps)

            # 6. 角度提取 — 使用原始（未镜像）关键点 + 手部关键点
            if result:
                # 恢复原始关键点（如果刚才被镜像了）
                raw_keypoints = {}
                if self._mirror_display:
                    raw_keypoints = self.pose_detector.mirror_keypoints(
                        result["keypoints"], w
                    )
                else:
                    raw_keypoints = result["keypoints"]
                # 使用增强版：带手部关键点的角度计算
                hand_lms = result.get("hand_landmarks")
                angles = self.angle_extractor.extract_with_hand(raw_keypoints, hand_lms)
            else:
                angles = None

            # 7. 滤波 → 映射 → 显示
            if angles is not None:
                limit_dict = self.mapper.get_angle_limits_dict() if self.mapper else None
                filtered_angles, hold_mask = self.motion_filter.update(
                    angles, angle_limits=limit_dict, dt_ms=33.0
                )

                # 映射（仅用于展示映射后的电机值）
                if self.mapper:
                    motor_cmds = self.mapper.map_angles(filtered_angles.tolist())
                else:
                    motor_cmds = {}

                # 在视频画面上叠加角度数值
                self._draw_angle_overlay(skeleton_frame, filtered_angles, hold_mask)

                # 更新角度面板
                self.angles_ready.emit(filtered_angles, hold_mask, motor_cmds, True)

                # 日志记录到文件
                self._log_angles_to_file(angles, filtered_angles, hold_mask)
            else:
                # 未检测到人体
                self.angles_ready.emit(None, None, None, False)

        self.log("处理线程已结束")

    def _draw_angle_overlay(self, frame, angles, hold_mask):
        """在视频画面右上角叠加电机角度数值（大字体版）"""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        _ = angles

        motor_cmds = {}
        if self.mapper:
            motor_cmds = self.mapper.map_angles(angles.tolist() if angles is not None else [0]*6)

        # 半透明背景（加高加宽）
        panel_w = 310
        line_h = 32
        pad_top = 12
        pad_bottom = 8
        total_h = pad_top + 6 * line_h + pad_bottom
        x0 = w - panel_w - 8
        cv2.rectangle(overlay, (x0, 8), (x0 + panel_w, 8 + total_h), (0, 0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        motor_labels = ["M1 肩上下", "M2 肩前后", "M3 肘屈伸",
                        "M4 前臂旋", "M5 腕Pitch", "M6 腕Roll"]

        for i in range(6):
            y = pad_top + i * line_h + 22
            is_hold = hold_mask is not None and i < len(hold_mask) and hold_mask[i]
            color = (50, 220, 220) if is_hold else (0, 255, 100)

            motor_id = f"M{i+1}"
            if motor_cmds and motor_id in motor_cmds:
                value = motor_cmds[motor_id]["value"]
            else:
                value = 0.0

            text = f"{motor_labels[i]}:  {value:+.1f}°"
            cv2.putText(frame, text, (x0 + 14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 底部手臂标
        arm_label = "R" if self.arm_combo.currentData() == "right" else "L"
        cv2.putText(frame, f"Arm: {arm_label}", (x0 + 14, pad_top + 6 * line_h + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # ══════════════════════════════════════
    #  GUI 更新（主线程）
    # ══════════════════════════════════════

    def _on_frame_ready(self, frame, skeleton_frame, fps):
        """更新视频画面"""
        self.video_widget.update_frame(skeleton_frame, fps=fps)
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def _on_angles_ready(self, angles, hold_mask, motor_cmds, detected):
        """更新角度面板——显示映射后的 M1~M6 电机值"""
        if detected and motor_cmds is not None:
            self.angle_panel.update_angles(
                motor_cmds=motor_cmds,
                hold_mask=hold_mask.tolist() if hold_mask is not None else None,
                detected=True,
            )
        else:
            self.angle_panel.update_angles(
                motor_cmds={}, hold_mask=[False]*6, detected=False
            )

    def _update_status_fps(self):
        """定时更新状态栏 FPS"""
        pass  # 已在 _on_frame_ready 中更新

    # ══════════════════════════════════════
    #  日志
    # ══════════════════════════════════════

    def log(self, message: str):
        """添加日志"""
        self.log_signal.emit(message)

    def _append_log(self, message: str):
        """追加日志到 UI"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.append(f"[{timestamp}] {message}")
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _log_angles_to_file(self, raw_angles, filtered_angles, hold_mask):
        """将角度数据写入日志文件"""
        if not self.config.get("logging", {}).get("angle_log_enabled", True):
            return

        os.makedirs(LOG_DIR, exist_ok=True)
        today = datetime.now().strftime("%Y%m%d")
        log_path = os.path.join(LOG_DIR, f"angle_log_{today}.txt")

        record = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "angles": {
                "theta_1": float(round(filtered_angles[0], 1)),
                "theta_2": float(round(filtered_angles[1], 1)),
                "theta_3": float(round(filtered_angles[2], 1)),
                "theta_4": float(round(filtered_angles[3], 1)),
                "theta_5": float(round(filtered_angles[4], 1)),
                "theta_6": float(round(filtered_angles[5], 1)),
            },
            "filtered": bool(np.any(hold_mask)),
        }

        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"角度日志写入失败: {e}")

    # ══════════════════════════════════════
    #  窗口事件
    # ══════════════════════════════════════

    def closeEvent(self, event):
        """窗口关闭事件"""
        self.log("正在退出...")
        self._running = False
        self._stop_event.set()

        if self.camera:
            self.camera.stop()
        if self.pose_detector:
            self.pose_detector.close()

        event.accept()
