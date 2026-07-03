"""
M05 - 网络通信模块 (TCP Client)
职责：TCP 客户端连接、JSON 指令编码/发送、心跳保活、自动重连
"""

import json
import socket
import threading
import time
import logging
from typing import Optional, Callable
from queue import Queue, Empty
from datetime import datetime

logger = logging.getLogger(__name__)


class TcpClient:
    """TCP 网络客户端"""

    # 连接状态枚举
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"

    def __init__(self, host: str = "127.0.0.1", port: int = 9556,
                 auto_reconnect: bool = True,
                 reconnect_interval: float = 2.0,
                 heartbeat_interval: float = 1.0):
        self.host = host
        self.port = port
        self.auto_reconnect = auto_reconnect
        self.reconnect_interval = reconnect_interval
        self.heartbeat_interval = heartbeat_interval

        self._socket: Optional[socket.socket] = None
        self._running = False
        self._status = self.DISCONNECTED
        self._seq = 0
        self._send_queue: Queue = Queue(maxsize=100)
        self._send_fail_count = 0

        # 回调
        self.on_status_change: Optional[Callable[[str], None]] = None
        self.on_ack: Optional[Callable[[dict], None]] = None

        # 线程
        self._send_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._reconnect_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ── 属性 ──

    @property
    def status(self) -> str:
        return self._status

    @property
    def is_connected(self) -> bool:
        return self._status == self.CONNECTED

    # ── 连接管理 ──

    def connect(self):
        """建立 TCP 连接"""
        self._running = True
        self._do_connect()

    def _do_connect(self):
        """实际连接逻辑"""
        self._set_status(self.CONNECTING)
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(3.0)
            self._socket.connect((self.host, self.port))
            self._socket.settimeout(None)
            self._send_fail_count = 0
            self._set_status(self.CONNECTED)
            logger.info(f"TCP 已连接: {self.host}:{self.port}")

            # 启动发送线程
            if self._send_thread is None or not self._send_thread.is_alive():
                self._send_thread = threading.Thread(
                    target=self._send_loop, name="TcpSend", daemon=True
                )
                self._send_thread.start()

            # 启动心跳线程
            if self.heartbeat_interval > 0:
                if self._heartbeat_thread is None or not self._heartbeat_thread.is_alive():
                    self._heartbeat_thread = threading.Thread(
                        target=self._heartbeat_loop, name="TcpHeartbeat", daemon=True
                    )
                    self._heartbeat_thread.start()

        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            logger.warning(f"连接失败: {e}")
            self._set_status(self.DISCONNECTED)
            if self.auto_reconnect and self._running:
                self._start_reconnect_timer()

    def disconnect(self):
        """断开连接"""
        self._running = False
        self._close_socket()
        self._set_status(self.DISCONNECTED)
        logger.info("TCP 已断开")

    def _close_socket(self):
        with self._lock:
            if self._socket:
                try:
                    self._socket.close()
                except Exception:
                    pass
                self._socket = None

    def _start_reconnect_timer(self):
        """启动重连定时器"""
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            return
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_loop, name="TcpReconnect", daemon=True
        )
        self._reconnect_thread.start()

    def _reconnect_loop(self):
        """自动重连循环"""
        self._set_status(self.RECONNECTING)
        while self._running and not self.is_connected:
            logger.info(f"自动重连中 ({self.reconnect_interval}s)...")
            time.sleep(self.reconnect_interval)
            if self._running:
                self._do_connect()

    # ── 状态管理 ──

    def _set_status(self, status: str):
        self._status = status
        if self.on_status_change:
            self.on_status_change(status)

    # ── 消息发送 ──

    def send_angle_cmd(self, motor_angles: dict, hold_mask: list = None):
        """
        发送角度指令

        Args:
            motor_angles: {"M1": 45.3, "M2": 12.7, ...}
            hold_mask: [False, False, False, False, False, False]
        """
        if hold_mask is None:
            hold_mask = [False] * 6

        self._seq += 1
        msg = {
            "type": "angle_cmd",
            "seq": self._seq,
            "timestamp": datetime.now().isoformat(timespec="microseconds"),
            "angles": motor_angles,
            "hold_mask": hold_mask,
        }
        self._enqueue_send(msg)

    def send_heartbeat(self):
        """发送心跳消息"""
        self._seq += 1
        msg = {
            "type": "heartbeat",
            "seq": self._seq,
            "timestamp": datetime.now().isoformat(timespec="microseconds"),
        }
        self._enqueue_send(msg)

    def _enqueue_send(self, msg: dict):
        """将消息放入发送队列"""
        try:
            self._send_queue.put_nowait(json.dumps(msg) + "\n")
        except Exception:
            logger.warning("发送队列已满，丢弃消息")

    # ── 发送线程 ──

    def _send_loop(self):
        """发送线程主循环"""
        buffer = ""
        while self._running:
            try:
                data = self._send_queue.get(timeout=0.1)
                if data:
                    self._send_raw(data)
                    self._send_fail_count = 0
            except Empty:
                continue
            except Exception as e:
                self._send_fail_count += 1
                logger.error(f"发送失败 ({self._send_fail_count}): {e}")
                if self._send_fail_count >= 3:
                    logger.warning("连续 3 次发送失败，标记断开")
                    self._set_status(self.DISCONNECTED)
                    self._close_socket()
                    if self.auto_reconnect:
                        self._start_reconnect_timer()
                    break

    def _send_raw(self, data: str):
        """发送原始数据"""
        with self._lock:
            if self._socket:
                self._socket.sendall(data.encode("utf-8"))

    def _heartbeat_loop(self):
        """心跳线程"""
        while self._running:
            if self.is_connected:
                self.send_heartbeat()
            time.sleep(self.heartbeat_interval)

    # ── 接收（可选） ──

    def start_listening(self):
        """启动接收线程（需要时调用）"""
        thread = threading.Thread(target=self._listen_loop, name="TcpListen", daemon=True)
        thread.start()

    def _listen_loop(self):
        """接收线程"""
        while self._running and self.is_connected:
            try:
                data = self._socket.recv(4096)
                if data:
                    for line in data.decode("utf-8").strip().split("\n"):
                        if line:
                            self._handle_ack(line)
            except (socket.timeout, OSError):
                break
            except Exception as e:
                logger.error(f"接收异常: {e}")
                break

    def _handle_ack(self, line: str):
        """处理应答消息"""
        try:
            msg = json.loads(line)
            if msg.get("type") == "ack":
                if self.on_ack:
                    self.on_ack(msg)
        except json.JSONDecodeError:
            pass

    def release(self):
        """释放资源"""
        self.disconnect()
