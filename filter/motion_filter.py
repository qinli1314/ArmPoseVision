"""
M03 - 姿态运动滤波模块 (Motion Filter)
职责：Dead Zone 静止保持 + EMA 指数移动平均 + 保持超时刷新
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MotionFilter:
    """姿态运动滤波器"""

    def __init__(self, dead_zone_deg: float = 5.0,
                 ema_alpha: float = 0.3,
                 hold_timeout_ms: int = 500,
                 n_dof: int = 6):
        """
        Args:
            dead_zone_deg: 死区阈值（°），变化小于此值保持
            ema_alpha: EMA 平滑系数 (0-1)，0 无平滑，1 不过滤
            hold_timeout_ms: 保持超时(ms)，超时后强制刷新
            n_dof: 自由度数量
        """
        self.dead_zone_deg = dead_zone_deg
        self.ema_alpha = ema_alpha
        self.hold_timeout_ms = hold_timeout_ms
        self.n_dof = n_dof

        # 状态
        self._last_sent = np.zeros(n_dof, dtype=np.float32)
        self._smoothed = np.zeros(n_dof, dtype=np.float32)
        self._hold_counters = np.zeros(n_dof, dtype=np.int32)
        self._initialized = False

    def reset(self):
        """重置滤波器状态"""
        self._last_sent = np.zeros(self.n_dof, dtype=np.float32)
        self._smoothed = np.zeros(self.n_dof, dtype=np.float32)
        self._hold_counters = np.zeros(self.n_dof, dtype=np.int32)
        self._initialized = False
        logger.debug("滤波器状态已重置")

    def update(self, raw_angles: np.ndarray,
               angle_limits: Optional[dict] = None,
               dt_ms: float = 33.0) -> tuple:
        """
        对输入角度进行滤波处理

        Args:
            raw_angles: shape (6,) 原始角度数组（度）
            angle_limits: 电机限位字典 {"M1": {"min": 0, "max": 180}, ...}
            dt_ms: 帧间隔时间(ms)，用于超时计算

        Returns:
            (output_angles, hold_mask)
            - output_angles: shape (6,) 输出角度
            - hold_mask: shape (6,) bool 数组，True=保持通道
        """
        if raw_angles is None or len(raw_angles) != self.n_dof:
            return self._last_sent.copy(), np.ones(self.n_dof, dtype=bool)

        current = np.asarray(raw_angles, dtype=np.float32)
        output = self._last_sent.copy()
        hold_mask = np.zeros(self.n_dof, dtype=bool)

        # 首次初始化
        if not self._initialized:
            self._last_sent = current.copy()
            self._smoothed = current.copy()
            self._initialized = True
            output = current.copy()
            return output, hold_mask

        # 逐通道滤波
        for i in range(self.n_dof):
            delta = abs(current[i] - self._last_sent[i])

            if delta < self.dead_zone_deg:
                # ── Dead Zone: 保持指令 ──
                self._hold_counters[i] += 1
                hold_time = self._hold_counters[i] * dt_ms

                if hold_time >= self.hold_timeout_ms:
                    # 超时强制刷新
                    output[i] = current[i]
                    self._last_sent[i] = current[i]
                    self._hold_counters[i] = 0
                    hold_mask[i] = False
                    logger.debug(f"通道 {i+1} 保持超时强制刷新: {current[i]:.1f}°")
                else:
                    # 保持上一帧值
                    output[i] = self._last_sent[i]
                    hold_mask[i] = True
            else:
                # ── EMA 平滑 ──
                self._smoothed[i] = (self.ema_alpha * current[i] +
                                     (1 - self.ema_alpha) * self._smoothed[i])
                output[i] = self._smoothed[i]
                self._last_sent[i] = self._smoothed[i]
                self._hold_counters[i] = 0
                hold_mask[i] = False

        # 角度钳位
        if angle_limits:
            output = self._clamp_angles(output, angle_limits)

        return output, hold_mask

    def _clamp_angles(self, angles: np.ndarray,
                      angle_limits: dict) -> np.ndarray:
        """将角度钳位到物理限位"""
        result = angles.copy()
        motor_keys = sorted(angle_limits.keys())  # M1, M2, ...
        for i, motor in enumerate(motor_keys):
            if i < len(result) and motor in angle_limits:
                limits = angle_limits[motor]
                result[i] = np.clip(result[i], limits["min"], limits["max"])
        return result

    @property
    def last_sent(self) -> np.ndarray:
        return self._last_sent.copy()

    def set_params(self, dead_zone_deg: float = None,
                   ema_alpha: float = None,
                   hold_timeout_ms: int = None):
        """运行时更新滤波参数"""
        if dead_zone_deg is not None:
            self.dead_zone_deg = dead_zone_deg
        if ema_alpha is not None:
            self.ema_alpha = ema_alpha
        if hold_timeout_ms is not None:
            self.hold_timeout_ms = hold_timeout_ms
        logger.info(f"滤波参数已更新: dead_zone={self.dead_zone_deg}°, "
                    f"ema_alpha={self.ema_alpha}, hold_timeout={self.hold_timeout_ms}ms")
