"""
M02 - 6DOF 关节角度计算模块 (Angle Extractor)
职责：基于人体关键点坐标计算肩/肘/腕的6个自由度角度

角度定义:
  θ₁ 肩上下 (Abduction) - 大臂相对于躯干的上下抬举，~0°=自然下垂，~90°=侧平举
  θ₂ 肩前后 (Flexion)   - 大臂相对于躯干的前后摆动，~0°=身侧，~90°=前平伸
  θ₃ 肘屈伸 (Flexion)   - 大臂与小臂之间的弯曲角度，0°=伸直，90°=直角
  θ₄ 前臂旋转 (Rotation) - 前臂绕自身轴的旋转（近似值）
  θ₅ 腕 Pitch           - 手腕上下弯曲（基于深度信息估算）
  θ₆ 腕 Roll            - 手腕左右旋转（需手部关键点）
"""

import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class AngleExtractor:
    """关节角度提取器"""

    def __init__(self, arm: str = "right"):
        self.arm = arm

    def extract(self, keypoints: dict) -> Optional[np.ndarray]:
        """
        从关键点坐标计算 6DOF 角度

        Args:
            keypoints: 包含 shoulder, elbow, wrist, hip, other_shoulder, other_hip
                       每个值为 (x, y, z) 元组

        Returns:
            shape (6,) 的 numpy 数组 [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆] (度)
            如果关键点不足返回 None
        """
        required = ["shoulder", "elbow", "wrist", "hip", "other_shoulder"]
        if not all(k in keypoints for k in required):
            return None

        # 提取向量
        shoulder = np.array(keypoints["shoulder"][:3], dtype=np.float64)
        elbow = np.array(keypoints["elbow"][:3], dtype=np.float64)
        wrist = np.array(keypoints["wrist"][:3], dtype=np.float64)
        hip = np.array(keypoints["hip"][:3], dtype=np.float64)
        other_shoulder = np.array(keypoints["other_shoulder"][:3], dtype=np.float64)

        # ── 构建坐标系 ──

        # 躯干向量 (髋→肩)
        torso = shoulder - hip
        torso_norm = np.linalg.norm(torso)
        if torso_norm < 1e-6:
            torso = np.array([0, -1, 0], dtype=np.float64)
            torso_n = torso
        else:
            torso_n = torso / torso_norm

        # 肩膀向量 (左→右)
        shoulder_vec = np.array(keypoints["other_shoulder"][:3], dtype=np.float64) - shoulder
        sv_norm = np.linalg.norm(shoulder_vec)
        if sv_norm < 1e-6:
            shoulder_vec = np.array([1, 0, 0], dtype=np.float64)
            shoulder_n = shoulder_vec
        else:
            shoulder_n = shoulder_vec / sv_norm

        # 侧向轴 = 肩膀 × 躯干 (指向身体前方)
        lateral = np.cross(shoulder_n, torso_n)
        lat_norm = np.linalg.norm(lateral)
        if lat_norm > 1e-6:
            lateral_n = lateral / lat_norm
        else:
            lateral_n = np.array([0, 0, 1], dtype=np.float64)

        # 大臂向量 (肩→肘)
        upper_arm = elbow - shoulder
        ua_norm = np.linalg.norm(upper_arm)
        if ua_norm < 1e-6:
            return None
        upper_n = upper_arm / ua_norm

        # 小臂向量 (肘→腕)
        forearm = wrist - elbow
        fa_norm = np.linalg.norm(forearm)
        if fa_norm < 1e-6:
            return None
        forearm_n = forearm / fa_norm

        # ============================================================
        # θ₁: 肩上下 (Abduction/Adduction)
        # ============================================================
        # 大臂在"躯干-侧向"平面（即身体的冠状面）上的投影与躯干的夹角
        # 0°=手臂下垂, 90°=侧平举, 180°=手臂上举过头

        # 投影到冠状面（去掉肩膀方向的分量）
        upper_coronal = upper_arm - np.dot(upper_arm, shoulder_n) * shoulder_n
        uc_norm = np.linalg.norm(upper_coronal)
        if uc_norm > 1e-6:
            upper_coronal_n = upper_coronal / uc_norm
            # 与躯干方向的夹角（取锐角）
            cos_t1 = np.clip(np.dot(upper_coronal_n, torso_n), -1.0, 1.0)
            raw_angle = np.degrees(np.arccos(cos_t1))
            # 转换为0-180: 0°=自然下垂, 180°=举过头顶
            theta_1 = 180.0 - raw_angle
            # 钳位
            theta_1 = np.clip(theta_1, 0.0, 180.0)
        else:
            theta_1 = 0.0

        # ============================================================
        # θ₂: 肩前后 (Flexion/Extension)
        # ============================================================
        # 大臂绕躯干轴的前后摆动角度
        # 将大臂投影到水平面（垂直于躯干），计算与侧向轴的夹角
        # 0°=手臂在身侧, 90°=前平伸

        horizontal = upper_arm - np.dot(upper_arm, torso_n) * torso_n
        h_norm = np.linalg.norm(horizontal)
        if h_norm > 1e-6:
            horizontal_n = horizontal / h_norm
            cos_t2 = np.clip(np.dot(horizontal_n, lateral_n), -1.0, 1.0)
            theta_2 = np.degrees(np.arccos(cos_t2))

            # 判断前后方向：用"前"向量（肩膀×躯干）点积判断
            forward = np.cross(shoulder_n, torso_n)
            f_norm = np.linalg.norm(forward)
            if f_norm > 1e-6:
                forward_n = forward / f_norm
                if np.dot(horizontal_n, forward_n) < 0:
                    theta_2 = -theta_2  # 手臂向后摆 → 负值
        else:
            theta_2 = 0.0

        # ============================================================
        # θ₃: 肘屈伸 (Flexion/Extension)
        # ============================================================
        # 大臂和小臂之间的夹角
        # 0°=完全伸直, 90°=直角, 150°=完全弯曲
        cos_t3 = np.clip(np.dot(upper_n, forearm_n), -1.0, 1.0)
        theta_3 = np.degrees(np.arccos(cos_t3))

        # ============================================================
        # θ₄: 前臂旋转 (Pronation/Supination)
        # ============================================================
        # 计算肘关节平面的法向量相对于参考方向的角度
        # 近似计算，精确需要手部关键点
        elbow_flex_normal = np.cross(upper_n, forearm_n)
        efn_norm = np.linalg.norm(elbow_flex_normal)
        if efn_norm > 1e-6 and fa_norm > 1e-6:
            elbow_flex_normal = elbow_flex_normal / efn_norm

            # 将法向量投影到垂直于小臂的平面
            proj = elbow_flex_normal - np.dot(elbow_flex_normal, forearm_n) * forearm_n
            pn_norm = np.linalg.norm(proj)
            if pn_norm > 1e-6:
                proj_n = proj / pn_norm
                # 参考方向：垂直于小臂和躯干
                ref_dir = np.cross(forearm_n, torso_n)
                rd_norm = np.linalg.norm(ref_dir)
                if rd_norm > 1e-6:
                    ref_dir_n = ref_dir / rd_norm
                    cos_t4 = np.clip(np.dot(proj_n, ref_dir_n), -1.0, 1.0)
                    theta_4 = np.degrees(np.arccos(cos_t4))

                    # 符号判断
                    vert_check = np.dot(proj_n, torso_n)
                    if vert_check < 0:
                        theta_4 = -theta_4
                else:
                    theta_4 = 0.0
            else:
                theta_4 = 0.0
        else:
            theta_4 = 0.0

        # ============================================================
        # θ₅: 腕 Pitch (上下弯曲)
        # ============================================================
        # 从 MediaPipe 的深度信息估算手腕上下弯曲
        # 原理：手弯曲时，腕关节的 z(深度) 会相对肘关节发生变化
        # 使用小臂向量在图像平面法线方向的偏移来估算
        # 注意：单目深度信息不精确，此值为大致估算

        # 方法：计算小臂向量偏离图像平面的角度
        # 在 MediaPipe 中，z 表示关键点在相机空间中的深度
        # 正 z = 朝向相机
        fa_xy = np.sqrt(forearm[0]**2 + forearm[1]**2 + 1e-8)
        theta_5 = np.degrees(np.arctan2(abs(forearm[2]), fa_xy))

        # 符号：根据手腕相对于肘关节的深度方向
        if abs(forearm[2]) > 1e-6:
            # 手心向上或向下会影响 z 方向
            # 结合前臂旋转角度来判断方向
            theta_5 = np.clip(theta_5, -90.0, 90.0)

        # ============================================================
        # θ₆: 腕 Roll (左右旋转)
        # ============================================================
        # 需要手部关键点才能精确计算
        # 这里的估算基于腕关节相对位置的微小变化
        # 方法：使用 other_shoulder→shoulder 方向作为参考
        # 结合 wrist 相对于 elbow 的横向偏移
        # 这个估算非常粗略，建议后续接入 MediaPipe Hands 改进

        # 计算肘→腕向量与肩宽向量的夹角变化
        forearm_horiz = forearm - np.dot(forearm, torso_n) * torso_n
        fh_norm = np.linalg.norm(forearm_horiz)
        if fh_norm > 1e-6 and sv_norm > 1e-6:
            forearm_horiz_n = forearm_horiz / fh_norm
            cos_t6 = np.clip(np.dot(forearm_horiz_n, shoulder_n), -1.0, 1.0)
            theta_6 = 90.0 - np.degrees(np.arccos(cos_t6))
        else:
            theta_6 = 0.0

        angles = np.array([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6],
                          dtype=np.float32)

        return angles

    def extract_with_hand(self, keypoints: dict, hand_landmarks=None) -> np.ndarray:
        """
        带手部关键点的角度计算（增强版）
        当检测到手部关键点时提供更精确的 θ₅, θ₆

        Args:
            keypoints: 上肢关键点
            hand_landmarks: MediaPipe 手部关键点列表

        Returns:
            shape (6,) 的角度数组
        """
        angles = self.extract(keypoints)
        if angles is None:
            return np.zeros(6, dtype=np.float32)

        if hand_landmarks is not None and len(hand_landmarks) > 0:
            try:
                wrist_pos = np.array(keypoints["wrist"][:3])
                forearm_vec = np.array(keypoints["elbow"][:3]) - wrist_pos
                forearm_n = forearm_vec / (np.linalg.norm(forearm_vec) + 1e-8)

                if len(hand_landmarks) >= 21:
                    # 取中指根部 (9) 和指尖 (12)
                    mid_prox = np.array([
                        hand_landmarks[9].x,
                        hand_landmarks[9].y,
                        hand_landmarks[9].z
                    ])
                    mid_tip = np.array([
                        hand_landmarks[12].x,
                        hand_landmarks[12].y,
                        hand_landmarks[12].z
                    ])

                    # 手掌方向
                    hand_dir = mid_tip - mid_prox
                    hd_norm = np.linalg.norm(hand_dir)
                    if hd_norm > 1e-6:
                        hand_dir_n = hand_dir / hd_norm
                        pitch_angle = np.degrees(np.arccos(
                            np.clip(np.dot(hand_dir_n, forearm_n), -1.0, 1.0)
                        ))
                        angles[4] = 90.0 - pitch_angle

                    # θ₆: 手掌绕前臂轴的旋转
                    index_base = np.array([
                        hand_landmarks[5].x,
                        hand_landmarks[5].y,
                        hand_landmarks[5].z
                    ])
                    pinky_base = np.array([
                        hand_landmarks[17].x,
                        hand_landmarks[17].y,
                        hand_landmarks[17].z
                    ])
                    hand_plane = index_base - pinky_base
                    hp_norm = np.linalg.norm(hand_plane)
                    if hp_norm > 1e-6:
                        hand_plane_n = hand_plane / hp_norm
                        roll_axis = np.cross(forearm_n, hand_plane_n)
                        ra_norm = np.linalg.norm(roll_axis)
                        if ra_norm > 1e-6:
                            roll_axis_n = roll_axis / ra_norm
                            cos_t6 = np.clip(np.dot(roll_axis_n, hand_dir_n)
                                             if hd_norm > 0 else 0, -1.0, 1.0)
                            angles[5] = np.degrees(np.arccos(cos_t6))
            except Exception as e:
                logger.warning(f"手部角度计算异常: {e}")

        return angles

    def set_arm(self, arm: str):
        """设置追踪手臂"""
        self.arm = arm
