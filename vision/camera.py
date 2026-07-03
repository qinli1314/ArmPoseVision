"""
M01 - 摄像头采集模块 (Camera Capture Module)
职责：枚举 USB 摄像头、启停采集、帧率控制、图像预处理（镜像翻转）
支持：USB 摄像头实时采集 / 视频文件回放
"""

import cv2
import numpy as np
import time
import logging
import os
from typing import Optional, List, Tuple
from threading import Thread, Event
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class CameraCapture:
    """摄像头/视频文件采集封装，运行在独立线程中"""

    def __init__(self, device_id: int = 0, resolution: Tuple[int, int] = (1280, 720),
                 target_fps: int = 30, mirror: bool = True,
                 video_path: Optional[str] = None):
        """
        Args:
            device_id: USB 摄像头设备 ID（video_path=None 时使用）
            resolution: 目标分辨率 (width, height)
            target_fps: 目标帧率
            mirror: 是否镜像翻转
            video_path: 视频文件路径（不为 None 时从文件读取）
        """
        self.device_id = device_id
        self.target_size = resolution
        self.target_fps = target_fps
        self.mirror = mirror
        self.video_path = video_path
        self.is_video_file = video_path is not None

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[Thread] = None
        self._stop_event = Event()
        self.frame_queue: Queue = Queue(maxsize=3)
        self._current_fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.perf_counter()
        self._total_frames = 0  # 视频总帧数
        self._loop_video = True  # 视频文件是否循环播放

    # ── 工厂方法 ──────────────────────────────────

    @classmethod
    def from_video(cls, video_path: str, mirror: bool = True,
                   loop: bool = True) -> "CameraCapture":
        """
        从视频文件创建采集器

        Args:
            video_path: 视频文件路径（支持 mp4, avi, mov 等）
            mirror: 是否镜像翻转
            loop: 是否循环播放
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        # 先打开获取基本信息
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        instance = cls(
            device_id=0,
            resolution=(w, h),
            target_fps=max(1, int(fps)),
            mirror=mirror,
            video_path=video_path,
        )
        instance._total_frames = total
        instance._loop_video = loop
        logger.info(f"视频文件加载: {os.path.basename(video_path)} "
                    f"{w}x{h} @ {fps:.1f}fps, {total} 帧, "
                    f"循环={'是' if loop else '否'}")
        return instance

    # ── 摄像头枚举 ──────────────────────────────────

    @staticmethod
    def list_cameras(max_devices: int = 8) -> List[int]:
        """扫描系统可用的 USB 摄像头设备"""
        available = []
        for i in range(max_devices):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
            cap.release()
        return available

    @staticmethod
    def list_cameras_with_name(max_devices: int = 8) -> List[dict]:
        """扫描摄像头并返回带名称的列表"""
        available = []
        for i in range(max_devices):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    name = cap.getBackendName()
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    available.append({
                        "id": i,
                        "name": f"Camera {i} ({name})",
                        "resolution": (w, h)
                    })
            cap.release()
        return available

    # ── 采集控制 ──────────────────────────────────

    def start(self):
        """启动采集/回放线程"""
        if self._running:
            logger.warning("已在运行")
            return

        if self.video_path:
            # 从视频文件读取
            self._cap = cv2.VideoCapture(self.video_path)
            if not self._cap.isOpened():
                raise RuntimeError(f"无法打开视频文件: {self.video_path}")
            logger.info(f"视频文件已打开: {os.path.basename(self.video_path)}")
        else:
            # 从 USB 摄像头读取
            self._cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                raise RuntimeError(f"无法打开摄像头设备 {self.device_id}")

            # 设置分辨率
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_size[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_size[1])
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"摄像头已打开: 设备={self.device_id}, 分辨率={actual_w}x{actual_h}")

        self._running = True
        self._stop_event.clear()
        self._thread = Thread(target=self._capture_loop, name="CaptureThread", daemon=True)
        self._thread.start()
        logger.info("采集线程已启动")

    def stop(self):
        """停止采集/回放"""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        logger.info("采集线程已停止")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps(self) -> float:
        return self._current_fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def source_name(self) -> str:
        """返回当前源名称（用于显示）"""
        if self.video_path:
            return os.path.basename(self.video_path)
        return f"Camera {self.device_id}"

    # ── 内部循环 ──────────────────────────────────

    def _capture_loop(self):
        """采集/回放线程主循环"""
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                logger.error("采集源断开")
                break

            ret, frame = self._cap.read()
            if not ret:
                # 视频文件播放完毕
                if self.is_video_file and self._loop_video:
                    logger.info("视频播放完毕，重新开始循环")
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logger.info("视频播放完毕" if self.is_video_file else "读取帧失败")
                    break

            # FPS 统计
            self._frame_count += 1
            elapsed = time.perf_counter() - self._fps_timer
            if elapsed >= 1.0:
                self._current_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = time.perf_counter()

            # 放入队列（非阻塞，如果队列满则丢弃旧帧）
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
            self.frame_queue.put(frame)

        self._cap.release()
        logger.info("采集循环退出")

    def read_frame(self) -> Optional[np.ndarray]:
        """非阻塞读取最新一帧"""
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None

    def read_frame_blocking(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """阻塞读取一帧"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None

    def release(self):
        """释放资源"""
        self.stop()
