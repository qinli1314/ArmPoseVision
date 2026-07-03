"""
GUI - 角度数值面板 (Angle Panel)
职责：实时显示 6 个电机角度值（经映射表转换后的 M1~M6），带进度条
范围从 mapping_config.json 读取
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QFrame, QProgressBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
import logging
import json
import os

logger = logging.getLogger(__name__)

# 电机名称和关联的角度
MOTOR_INFO = [
    ("M1", "θ₁ 肩上下"),
    ("M2", "θ₂ 肩前后"),
    ("M3", "θ₃ 肘屈伸"),
    ("M4", "θ₄ 前臂旋"),
    ("M5", "θ₅ 腕Pitch"),
    ("M6", "θ₆ 腕Roll"),
]


def _load_motor_limits() -> dict:
    """从 mapping_config.json 读取电机限位"""
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "mapping_config.json"
    )
    defaults = {
        "M1": {"min": 0, "max": 180},
        "M2": {"min": -90, "max": 90},
        "M3": {"min": 0, "max": 150},
        "M4": {"min": -90, "max": 90},
        "M5": {"min": -60, "max": 60},
        "M6": {"min": -45, "max": 45},
    }
    if not os.path.exists(cfg_path):
        return defaults
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        limits = cfg.get("angle_limits", {})
        for k in defaults:
            if k not in limits:
                limits[k] = defaults[k]
        return limits
    except Exception:
        return defaults


class MotorBar(QWidget):
    """带进度条的单个电机角度组件"""

    def __init__(self, motor_id: str, angle_label: str,
                 range_min: float, range_max: float, parent=None):
        super().__init__(parent)
        self.motor_id = motor_id
        self.range_min = range_min
        self.range_max = range_max
        self.range_span = range_max - range_min

        self.setStyleSheet("""
            MotorBar {
                background-color: #252525;
                border: 1px solid #3a3a3a;
                border-radius: 6px;
            }
        """)
        self.setFixedHeight(66)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        # 电机编号 (固定宽)
        self.label = QLabel(motor_id)
        self.label.setFixedWidth(40)
        self.label.setStyleSheet("color: #ffcc88; font-size: 18px; font-weight: bold;")
        layout.addWidget(self.label)

        # 关联角度名
        self.angle_label = QLabel(angle_label)
        self.angle_label.setFixedWidth(130)
        self.angle_label.setStyleSheet("color: #eeeeee; font-size: 16px;")
        layout.addWidget(self.angle_label)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(50)
        self.progress.setFixedHeight(22)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                background-color: #1a1a1a;
                border: 1px solid #555;
                border-radius: 10px;
            }
            QProgressBar::chunk {
                background-color: #00cc88;
                border-radius: 10px;
            }
        """)
        layout.addWidget(self.progress, stretch=1)

        # 数值
        self.value_label = QLabel("--.-°")
        self.value_label.setFixedWidth(90)
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setStyleSheet("color: #00ff64; font-size: 22px; font-weight: bold;")
        layout.addWidget(self.value_label)

    def update_value(self, value: float, is_hold: bool = False):
        """更新角度值和进度条"""
        self.value_label.setText(f"{value:+.1f}°")

        # 进度条 0~100
        pct = int((value - self.range_min) / self.range_span * 100)
        pct = max(0, min(100, pct))
        self.progress.setValue(pct)

        bar_color = "#ffc800" if is_hold else "#00cc88"
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: #1a1a1a;
                border: 1px solid #555;
                border-radius: 10px;
            }}
            QProgressBar::chunk {{
                background-color: {bar_color};
                border-radius: 10px;
            }}
        """)
        self.value_label.setStyleSheet(
            f"color: {'#ffc800' if is_hold else '#00ff64'}; font-size: 22px; font-weight: bold;"
        )


class AnglePanel(QWidget):
    """角度数值面板 — 显示映射后的 M1~M6 电机值"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(320)
        self.setMaximumWidth(400)

        # 从配置文件读取限位
        self.motor_limits = _load_motor_limits()

        layout = QVBoxLayout(self)
        layout.setSpacing(3)
        layout.setContentsMargins(6, 6, 6, 6)

        # ── 标题栏 ──
        header = QHBoxLayout()
        title = QLabel("⛭ 电机角度")
        title.setStyleSheet("color: #ffffff; font-size: 18px; font-weight: bold;")
        header.addWidget(title)

        self.arm_label = QLabel("🔄 右臂")
        self.arm_label.setStyleSheet(
            "color: #88ccff; font-size: 14px; padding: 2px 10px;"
            "border: 1px solid #88ccff; border-radius: 8px;"
        )
        header.addStretch()
        header.addWidget(self.arm_label)
        layout.addLayout(header)

        # ── 6 个电机条 ──
        self.bars = []
        for motor_id, angle_label in MOTOR_INFO:
            limits = self.motor_limits.get(motor_id, {"min": -180, "max": 180})
            bar = MotorBar(motor_id, angle_label, limits["min"], limits["max"])
            self.bars.append(bar)
            layout.addWidget(bar)

        # ── 底部状态 ──
        footer = QHBoxLayout()
        self.detection_label = QLabel("● 无人")
        self.detection_label.setStyleSheet("color: #888; font-size: 13px;")
        footer.addWidget(self.detection_label)

        footer.addStretch()
        self.mode_label = QLabel("📋 映射后值")
        self.mode_label.setStyleSheet("color: #666; font-size: 13px;")
        footer.addWidget(self.mode_label)
        layout.addLayout(footer)

    def set_arm(self, arm: str):
        text = "🔄 右臂" if arm == "right" else "🔄 左臂"
        self.arm_label.setText(text)

    def update_angles(self, motor_cmds: dict = None,
                      hold_mask: list = None, detected: bool = True):
        """
        更新所有电机角度显示

        Args:
            motor_cmds: {"M1": {"value": 45.3}, "M2": ...}
            hold_mask: [bool, ...]
            detected: 是否检测到人体
        """
        if hold_mask is None:
            hold_mask = [False] * 6

        # 检测状态
        if detected:
            self.detection_label.setText("● 已检测")
            self.detection_label.setStyleSheet("color: #00cc88; font-size: 13px; font-weight: bold;")
        else:
            self.detection_label.setText("● 无人")
            self.detection_label.setStyleSheet("color: #888; font-size: 13px;")

        # 更新每个电机
        for i, bar in enumerate(self.bars):
            motor_id = bar.motor_id
            is_hold = i < len(hold_mask) and hold_mask[i]

            if motor_cmds and motor_id in motor_cmds:
                value = motor_cmds[motor_id]["value"]
                bar.update_value(value, is_hold)
            else:
                bar.update_value(0, is_hold)

    def reload_limits(self):
        """热重载限位（配置文件变更时调用）"""
        self.motor_limits = _load_motor_limits()
        for i, bar in enumerate(self.bars):
            motor_id = bar.motor_id
            limits = self.motor_limits.get(motor_id, {"min": -180, "max": 180})
            bar.range_min = limits["min"]
            bar.range_max = limits["max"]
            bar.range_span = limits["max"] - limits["min"]
