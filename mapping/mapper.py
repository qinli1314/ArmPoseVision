"""
M04 - 角度映射模块 (Mapper)
职责：6D 角度向量 → M1–M6 电机编号映射，支持配置文件热重载
"""

import json
import os
import logging
from typing import Optional, List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MappingEntry:
    """单条映射配置"""
    angle_index: int      # 1-6
    motor_id: str         # "M1"-"M6"
    scale: float = 1.0    # 角度缩放系数
    offset: float = 0.0   # 角度偏移量
    invert: bool = False  # 是否反转方向


class Mapper:
    """角度→电机映射器"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "..", "config", "mapping_config.json"
        )
        self.config_path = os.path.abspath(self.config_path)

        self.mappings: List[MappingEntry] = []
        self.angle_limits: Dict[str, dict] = {}
        self._mtime = 0  # 文件修改时间

        self.load_config()

    def load_config(self) -> bool:
        """加载映射配置文件"""
        if not os.path.exists(self.config_path):
            logger.error(f"映射配置文件不存在: {self.config_path}")
            self._load_defaults()
            return False

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            self.mappings = []
            for item in config.get("mappings", []):
                self.mappings.append(MappingEntry(
                    angle_index=item["angle_index"],
                    motor_id=item["motor_id"],
                    scale=item.get("scale", 1.0),
                    offset=item.get("offset", 0.0),
                    invert=item.get("invert", False),
                ))

            self.angle_limits = config.get("angle_limits", {})

            # 按 angle_index 排序
            self.mappings.sort(key=lambda m: m.angle_index)

            self._mtime = os.path.getmtime(self.config_path)
            logger.info(f"映射配置已加载: {len(self.mappings)} 条映射, "
                        f"{len(self.angle_limits)} 个限位")
            return True

        except Exception as e:
            logger.error(f"加载映射配置失败: {e}")
            self._load_defaults()
            return False

    def _load_defaults(self):
        """加载默认映射"""
        self.mappings = [
            MappingEntry(1, "M1"), MappingEntry(2, "M2"),
            MappingEntry(3, "M3"), MappingEntry(4, "M4"),
            MappingEntry(5, "M5"), MappingEntry(6, "M6"),
        ]
        self.angle_limits = {
            "M1": {"min": 0, "max": 180},
            "M2": {"min": -90, "max": 90},
            "M3": {"min": 0, "max": 150},
            "M4": {"min": -90, "max": 90},
            "M5": {"min": -60, "max": 60},
            "M6": {"min": -45, "max": 45},
        }

    def check_hot_reload(self) -> bool:
        """检查配置是否已更新，是则热重载"""
        if not os.path.exists(self.config_path):
            return False
        new_mtime = os.path.getmtime(self.config_path)
        if new_mtime > self._mtime:
            logger.info("检测到映射配置文件变更，执行热重载")
            return self.load_config()
        return False

    def map_angles(self, angles: List[float]) -> Dict[str, dict]:
        """
        将 6D 角度向量映射为电机指令

        Args:
            angles: [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆]

        Returns:
            {"M1": {"value": 45.3, "limit_min": 0, "limit_max": 180}, ...}
        """
        result = {}
        for mapping in self.mappings:
            idx = mapping.angle_index - 1
            if idx < 0 or idx >= len(angles):
                continue

            raw = angles[idx]

            # scale
            mapped = raw * mapping.scale

            # offset
            mapped += mapping.offset

            # invert
            if mapping.invert:
                mapped = -mapped

            # limit
            limits = self.angle_limits.get(mapping.motor_id, {})
            limit_min = limits.get("min", -180)
            limit_max = limits.get("max", 180)
            clamped = max(limit_min, min(limit_max, mapped))

            result[mapping.motor_id] = {
                "value": round(clamped, 1),
                "limit_min": limit_min,
                "limit_max": limit_max,
            }

        return result

    def get_angle_limits_dict(self) -> Dict[str, dict]:
        """获取角度限位字典（供滤波模块使用）"""
        return self.angle_limits
