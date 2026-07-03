"""
M02 - 姿态估计推理封装 (Pose Detection Module)
职责：封装 MediaPipe Pose + Hands 模型，提取人体上肢+手部关键点坐标
"""

import mediapipe as mp
import numpy as np
import cv2
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# MediaPipe Pose 关键点索引
KEYPOINT_INDICES = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}


class PoseDetector:
    """MediaPipe Pose + Hands 检测器封装"""

    def __init__(self, detection_confidence: float = 0.5,
                 tracking_confidence: float = 0.5,
                 arm: str = "right",
                 enable_hands: bool = True):
        """
        Args:
            detection_confidence: 检测置信度阈值
            tracking_confidence: 追踪置信度阈值
            arm: 追踪的手臂 ("left" 或 "right")
            enable_hands: 是否启用手部检测（用于精确计算腕关节角度）
        """
        self.arm = arm
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.enable_hands = enable_hands

        # 初始化 MediaPipe Pose
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        # 初始化 MediaPipe Hands（可选，用于腕关节角度精确计算）
        self._hands = None
        if enable_hands:
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("手部关键点检测已启用")

        self._update_keypoints()

    def _update_keypoints(self):
        """根据追踪手臂更新关键点索引"""
        if self.arm == "right":
            self.shoulder = KEYPOINT_INDICES["right_shoulder"]
            self.elbow = KEYPOINT_INDICES["right_elbow"]
            self.wrist = KEYPOINT_INDICES["right_wrist"]
            self.hip = KEYPOINT_INDICES["right_hip"]
            self.other_shoulder = KEYPOINT_INDICES["left_shoulder"]
            self.other_hip = KEYPOINT_INDICES["left_hip"]
            self._hand_label = "Right"
        else:
            self.shoulder = KEYPOINT_INDICES["left_shoulder"]
            self.elbow = KEYPOINT_INDICES["left_elbow"]
            self.wrist = KEYPOINT_INDICES["left_wrist"]
            self.hip = KEYPOINT_INDICES["left_hip"]
            self.other_shoulder = KEYPOINT_INDICES["right_shoulder"]
            self.other_hip = KEYPOINT_INDICES["right_hip"]
            self._hand_label = "Left"

    def switch_arm(self, arm: str):
        """切换追踪手臂"""
        self.arm = arm
        self._update_keypoints()
        logger.info(f"切换追踪手臂: {arm}")

    def detect(self, image: np.ndarray, mirror_hand: bool = False) -> Optional[dict]:
        """
        对单帧图像进行姿态检测+手部检测

        Args:
            image: BGR 图像 (H, W, 3)
            mirror_hand: 画面是否镜像（前置摄像头需设为 True，因为画面左右翻转，
                         MediaPipe 的手部左右标签会相反）

        Returns:
            检测结果字典，包含:
            - "keypoints": 提取的上肢关键点
            - "hand_landmarks": 手部 21 个关键点列表（若启用），或 None
            检测不到上肢时返回 None
        """
        if image is None:
            return None

        # MediaPipe 需要 RGB
        if image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image

        # ── 姿态检测 ──
        pose_results = self._pose.process(rgb)
        if not pose_results.pose_landmarks:
            return None

        landmarks = pose_results.pose_landmarks.landmark
        h, w = image.shape[:2]

        keypoints = {
            "shoulder": self._lm_to_coord(landmarks[self.shoulder], w, h),
            "elbow": self._lm_to_coord(landmarks[self.elbow], w, h),
            "wrist": self._lm_to_coord(landmarks[self.wrist], w, h),
            "hip": self._lm_to_coord(landmarks[self.hip], w, h),
            "other_shoulder": self._lm_to_coord(landmarks[self.other_shoulder], w, h),
            "other_hip": self._lm_to_coord(landmarks[self.other_hip], w, h),
        }

        result = {
            "keypoints": keypoints,
            "hand_landmarks": None,
        }

        # ── 手部检测（可选） ──
        if self._hands is not None:
            try:
                hand_results = self._hands.process(rgb)
                if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                    # 找到追踪手臂对应的手
                    # 前置摄像头画面镜像 -> 手的左右标签会互换
                    target_label = self._hand_label  # "Left" or "Right"
                    if mirror_hand:
                        target_label = "Left" if self._hand_label == "Right" else "Right"

                    for hand_lms, handedness in zip(
                        hand_results.multi_hand_landmarks,
                        hand_results.multi_handedness
                    ):
                        label = handedness.classification[0].label  # "Left" or "Right"
                        if label == target_label:
                            result["hand_landmarks"] = hand_lms.landmark
                            break
            except Exception as e:
                logger.warning(f"手部检测异常: {e}")

        return result

    @staticmethod
    def _lm_to_coord(landmark, width: int, height: int) -> Tuple[float, float, float]:
        """将 MediaPipe 关键点转为 (x, y, z) 像素坐标"""
        return (landmark.x * width, landmark.y * height, landmark.z * width)

    @staticmethod
    def mirror_keypoints(keypoints: dict, image_width: int) -> dict:
        """镜像翻转关键点 x 坐标"""
        mirrored = {}
        for name, (x, y, z) in keypoints.items():
            mirrored[name] = (image_width - x, y, z)
        return mirrored

    def draw_skeleton(self, image: np.ndarray, result: dict) -> np.ndarray:
        """
        在图像上绘制骨架和手部关键点

        Args:
            image: 原始 BGR 图像
            result: detect() 的返回值

        Returns:
            绘制了骨架的图像
        """
        if result is None:
            return image

        canvas = image.copy()
        kp = result["keypoints"]

        COLOR_TRUNK = (0, 255, 0)
        COLOR_ARM = (255, 0, 0)
        COLOR_UPPER_ARM = (0, 255, 255)
        COLOR_FOREARM = (255, 0, 255)
        COLOR_TRUNK_LINE = (255, 255, 255)
        COLOR_HAND = (0, 255, 255)

        pts = {name: (int(v[0]), int(v[1])) for name, v in kp.items()}

        # 骨架连线
        cv2.line(canvas, pts["shoulder"], pts["hip"], COLOR_TRUNK_LINE, 2)
        cv2.line(canvas, pts["shoulder"], pts["elbow"], COLOR_UPPER_ARM, 2)
        cv2.line(canvas, pts["elbow"], pts["wrist"], COLOR_FOREARM, 2)
        cv2.line(canvas, pts["shoulder"], pts["other_shoulder"], COLOR_TRUNK_LINE, 1)
        cv2.line(canvas, pts["hip"], pts["other_hip"], COLOR_TRUNK_LINE, 1)

        # 关键点
        cv2.circle(canvas, pts["shoulder"], 5, COLOR_TRUNK, -1)
        cv2.circle(canvas, pts["hip"], 5, COLOR_TRUNK, -1)
        cv2.circle(canvas, pts["other_shoulder"], 5, COLOR_TRUNK, -1)
        cv2.circle(canvas, pts["other_hip"], 5, COLOR_TRUNK, -1)
        cv2.circle(canvas, pts["elbow"], 5, COLOR_ARM, -1)
        cv2.circle(canvas, pts["wrist"], 5, COLOR_ARM, -1)

        # ── 绘制手部关键点 ──
        hand_lms = result.get("hand_landmarks")
        if hand_lms is not None and len(hand_lms) > 0:
            h, w = canvas.shape[:2]
            for i, lm in enumerate(hand_lms):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(canvas, (cx, cy), 3, COLOR_HAND, -1)

            # 画手掌连线（骨架）
            connections = mp.solutions.hands.HAND_CONNECTIONS
            for conn in connections:
                start = hand_lms[conn[0]]
                end = hand_lms[conn[1]]
                sx, sy = int(start.x * w), int(start.y * h)
                ex, ey = int(end.x * w), int(end.y * h)
                cv2.line(canvas, (sx, sy), (ex, ey), COLOR_HAND, 1)

        return canvas

    def close(self):
        """释放资源"""
        self._pose.close()
        if self._hands:
            self._hands.close()
        logger.info("姿态检测器已关闭")
