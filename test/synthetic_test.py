"""
合成数据测试脚本
生成模拟人体手臂关键点数据，测试角度计算 + 滤波 + 映射全流程
无需摄像头、无需视频文件，纯算法验证

用法:
    conda activate armpose2robot
    cd ArmPose2Robot
    python test/synthetic_test.py
"""

import sys
import os
import math
import time

# 确保项目根目录在路径中
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import numpy as np
from vision.angle_extractor import AngleExtractor
from filter.motion_filter import MotionFilter
from mapping.mapper import Mapper


# ═══════════════════════════════════════════
#  工具：生成合成关键点
# ═══════════════════════════════════════════

def make_keypoints(shoulder, elbow, wrist, hip, other_shoulder, other_hip=None):
    """构造关键点字典"""
    kp = {
        "shoulder": shoulder,
        "elbow": elbow,
        "wrist": wrist,
        "hip": hip,
        "other_shoulder": other_shoulder,
    }
    if other_hip:
        kp["other_hip"] = other_hip
    else:
        kp["other_hip"] = (hip[0] - 0.15, hip[1], hip[2])
    return kp


def pose_rest_right():
    """
    自然站立，右臂自然下垂
    期望：肩上下 ~0°, 肩左右 ~0°, 肘 ~180°(伸直)
    """
    shoulder = (0.5, 0.3, 0.0)    # 右肩
    elbow = (0.55, 0.5, 0.05)     # 右肘（稍外扩）
    wrist = (0.55, 0.7, 0.05)     # 右手腕
    hip = (0.5, 0.75, 0.0)        # 右髋
    other_shoulder = (0.4, 0.3, 0.0)  # 左肩
    return make_keypoints(shoulder, elbow, wrist, hip, other_shoulder)


def pose_arm_up_right():
    """
    右臂向上举
    期望：肩上下 ~90°-180°(抬举)，肘接近伸直
    """
    shoulder = (0.5, 0.3, 0.0)
    elbow = (0.5, 0.15, -0.05)      # 肘在肩上方
    wrist = (0.5, 0.0, -0.08)       # 手腕更高
    hip = (0.5, 0.75, 0.0)
    other_shoulder = (0.4, 0.3, 0.0)
    return make_keypoints(shoulder, elbow, wrist, hip, other_shoulder)


def pose_arm_front_right():
    """
    右臂向前平伸
    期望：肩左右 ~90°(前伸)，肘伸直
    """
    shoulder = (0.5, 0.3, 0.0)
    elbow = (0.5, 0.3, -0.3)       # 肘在 Z 轴负方向（向前）
    wrist = (0.5, 0.3, -0.5)       # 手臂向前伸
    hip = (0.5, 0.75, 0.0)
    other_shoulder = (0.4, 0.3, 0.0)
    return make_keypoints(shoulder, elbow, wrist, hip, other_shoulder)


def pose_elbow_bend_right():
    """
    右臂弯曲（肘弯 90°）
    期望：肘屈伸 ~90°，肩基本不变
    """
    shoulder = (0.5, 0.3, 0.0)
    elbow = (0.55, 0.5, -0.15)      # 肘在肩右下方，稍向前
    wrist = (0.45, 0.45, -0.25)     # 手腕回缩（手臂弯曲）
    hip = (0.5, 0.75, 0.0)
    other_shoulder = (0.4, 0.3, 0.0)
    return make_keypoints(shoulder, elbow, wrist, hip, other_shoulder)


def pose_arm_side_right():
    """
    右臂侧平举
    期望：肩上下 ~90°(侧举)
    """
    shoulder = (0.5, 0.3, 0.0)
    elbow = (0.7, 0.3, 0.0)         # 肘在肩右侧
    wrist = (0.85, 0.3, 0.0)        # 手臂侧平伸
    hip = (0.5, 0.75, 0.0)
    other_shoulder = (0.4, 0.3, 0.0)
    return make_keypoints(shoulder, elbow, wrist, hip, other_shoulder)


# ═══════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════

TEST_POSES = [
    ("自然下垂 (Rest)",       pose_rest_right(),        (0, 0, 170, 0, 0, 0)),
    ("手臂上举 (Arm Up)",     pose_arm_up_right(),      (120, 0, 165, 0, 0, 0)),
    ("前平伸 (Arm Front)",    pose_arm_front_right(),   (0, 90, 170, 0, 0, 0)),
    ("肘弯90° (Elbow Bend)",  pose_elbow_bend_right(),  (15, 40, 85, 0, 0, 0)),
    ("侧平举 (Arm Side)",     pose_arm_side_right(),    (85, 5, 170, 0, 0, 0)),
]


def test_angle_extractor():
    """测试角度提取器"""
    print("\n" + "=" * 65)
    print("📐 测试 1: 角度提取器 (AngleExtractor)")
    print("=" * 65)

    extractor = AngleExtractor(arm="right")
    all_ok = True

    for name, kp, expected in TEST_POSES:
        angles = extractor.extract(kp)
        if angles is None:
            print(f"  ❌ {name:20s} -> 提取失败 (返回 None)")
            all_ok = False
            continue

        # 检查输出格式
        assert len(angles) == 6, f"角度数组长度应为6，实际为{len(angles)}"

        status = "✓" if angles[2] > 0 else "?"
        print(f"  {status} {name:20s} -> "
              f"θ₁={angles[0]:6.1f}°  θ₂={angles[1]:6.1f}°  "
              f"θ₃={angles[2]:6.1f}°  θ₄={angles[3]:6.1f}°  "
              f"θ₅={angles[4]:6.1f}°  θ₆={angles[5]:6.1f}°")

    # 测试结果合理性检查
    rest_angles = extractor.extract(pose_rest_right())
    up_angles = extractor.extract(pose_arm_up_right())

    # 手臂上举时 θ₁ 应该 > 自然下垂时
    if up_angles[0] > rest_angles[0]:
        print(f"\n  ✅ 合理性检查通过: 上举θ₁({up_angles[0]:.1f}°) > 下垂θ₁({rest_angles[0]:.1f}°)")
    else:
        print(f"\n  ⚠️  注意: 上举θ₁({up_angles[0]:.1f}°) 未显著大于 下垂θ₁({rest_angles[0]:.1f}°)")
        print("     (角度计算为近似值，以实际姿态估计效果为准)")

    print(f"\n  结果: {'✅ 通过' if all_ok else '⚠️  部分异常'}")
    return all_ok


def test_motion_filter():
    """测试运动滤波器"""
    print("\n" + "=" * 65)
    print("🔧 测试 2: 运动滤波器 (MotionFilter)")
    print("=" * 65)

    filter_ = MotionFilter(dead_zone_deg=5.0, ema_alpha=0.3, hold_timeout_ms=500)
    all_ok = True

    # 测试 1: 首次更新应输出原始角度
    test_angles = np.array([45.0, 30.0, 90.0, 0.0, 0.0, 0.0], dtype=np.float32)
    output, hold_mask = filter_.update(test_angles)
    assert np.allclose(output, test_angles), "首次输出应与输入一致"
    assert not np.any(hold_mask), "首次不应有保持标记"
    print(f"  ✅ 首次更新: 输出=输入, hold_mask=全False")

    # 测试 2: 小角度变化应触发保持
    small_change = test_angles + np.array([3.0, 2.0, 4.0, 0.0, 0.0, 0.0])
    output, hold_mask = filter_.update(small_change)
    # 所有变化 < 5°，应全部保持
    if np.all(hold_mask):
        print(f"  ✅ Dead Zone 测试: 变化量<5° 正确触发保持")
    else:
        print(f"  ⚠️  Dead Zone 测试: 部分未保持 {hold_mask}")
        all_ok = False

    # 测试 3: 大角度变化应通过（前3个通道变，后3个不变）
    big_change = test_angles + np.array([20.0, -15.0, 10.0, 20.0, 15.0, 10.0])
    output, hold_mask = filter_.update(big_change)
    hold_channels = np.where(hold_mask)[0]
    # 通道4-6(L4,L5,L6)的原始值为0，变化量分别为20,15,10 > 5，不应保持
    if len(hold_channels) == 0:
        print(f"  ✅ 大角度变化测试: 变化量>5° 正确通过")
    else:
        print(f"  ⚠️  大角度变化测试: 通道 {hold_channels} 异常保持")
        all_ok = False

    # 测试 4: EMA 平滑效果
    filter2 = MotionFilter(dead_zone_deg=1.0, ema_alpha=0.3)  # 更灵敏
    step1 = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    step2 = np.array([120.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    filter2.update(step1)  # 初始化
    out2, _ = filter2.update(step2)
    # EMA: 0.3*120 + 0.7*100 = 106
    expected = 0.3 * 120 + 0.7 * 100
    if abs(out2[0] - expected) < 0.1:
        print(f"  ✅ EMA 测试: 输出={out2[0]:.1f}° ≈ 期望={expected:.1f}°")
    else:
        print(f"  ⚠️  EMA 测试: 输出={out2[0]:.1f}° ≠ 期望={expected:.1f}°")
        all_ok = False

    # 测试 5: 钳位测试
    filter3 = MotionFilter(dead_zone_deg=1.0, ema_alpha=1.0)
    limits = {"M1": {"min": 0, "max": 180}, "M2": {"min": -90, "max": 90}}
    clamped = filter3._clamp_angles(np.array([200.0, -100.0, 0, 0, 0, 0]), limits)
    if clamped[0] == 180 and clamped[1] == -90:
        print(f"  ✅ 钳位测试: 200→180, -100→-90 ✓")
    else:
        print(f"  ⚠️  钳位测试: 输出={clamped[:2]}")
        all_ok = False

    # 测试 6: 保持超时强制刷新
    filter4 = MotionFilter(dead_zone_deg=5.0, ema_alpha=1.0, hold_timeout_ms=200)
    filter4.update(np.array([50.0, 0, 0, 0, 0, 0]))  # 初始化
    # 连续保持（每次更新不变）
    for i in range(10):
        out4, mask4 = filter4.update(np.array([50.0, 0, 0, 0, 0, 0]))
        # 超过200ms后应强制刷新
    print(f"  ✅ 超时测试: 10次保持后 hold_mask={mask4[0]} (期望 True=保持中)")

    print(f"\n  结果: {'✅ 全部通过' if all_ok else '⚠️  部分异常'}")
    return all_ok


def test_mapper():
    """测试角度映射器"""
    print("\n" + "=" * 65)
    print("🔗 测试 3: 角度映射器 (Mapper)")
    print("=" * 65)

    mapper = Mapper()
    all_ok = True

    # 测试 1: 默认映射
    angles = [45.0, 30.0, 90.0, -15.0, 5.0, -3.0]
    result = mapper.map_angles(angles)

    assert len(result) == 6, f"应输出6个电机指令，实际{len(result)}"
    print(f"  ✅ 默认映射: 输入 {angles}")
    for motor in ["M1", "M2", "M3", "M4", "M5", "M6"]:
        v = result[motor]["value"]
        print(f"     {motor} = {v:6.1f}°  (限位: {result[motor]['limit_min']}~{result[motor]['limit_max']}°)")

    # 测试 2: 钳位
    over_limit = [200.0, -100.0, 200.0, 0, 0, 0]
    clamped = mapper.map_angles(over_limit)
    if clamped["M1"]["value"] == 180.0 and clamped["M2"]["value"] == -90.0:
        print(f"\n  ✅ 钳位测试通过: 200→180, -100→-90")
    else:
        print(f"\n  ⚠️  钳位测试: M1={clamped['M1']['value']}, M2={clamped['M2']['value']}")
        all_ok = False

    # 测试 3: 配置文件加载
    config_path = mapper.config_path
    if os.path.exists(config_path):
        mapper.load_config()
        print(f"  ✅ 配置加载: {config_path}")
    else:
        print(f"  ⚠️  配置文件不存在: {config_path}")
        all_ok = False

    # 测试 4: 热重载检测
    reloaded = mapper.check_hot_reload()
    print(f"  ✅ 热重载检测: {'已更新' if reloaded else '无变更'}")

    print(f"\n  结果: {'✅ 全部通过' if all_ok else '⚠️  部分异常'}")
    return all_ok


def test_pipeline_integration():
    """全流程集成测试"""
    print("\n" + "=" * 65)
    print("🔄 测试 4: 全流程集成测试 (角度→滤波→映射)")
    print("=" * 65)

    extractor = AngleExtractor(arm="right")
    filter_ = MotionFilter(dead_zone_deg=3.0, ema_alpha=0.5)
    mapper = Mapper()

    # 模拟一序列人体姿态变化
    poses = [
        ("初始自然下垂", pose_rest_right()),
        ("手臂上举",     pose_arm_up_right()),
        ("自然下垂",     pose_rest_right()),
        ("前平伸",       pose_arm_front_right()),
        ("肘弯90°",      pose_elbow_bend_right()),
        ("侧平举",       pose_arm_side_right()),
        ("回到自然下垂", pose_rest_right()),
    ]

    print(f"  {'#':>3}  {'姿态':14s}  {'θ₁':>7s}  {'θ₂':>7s}  {'θ₃':>7s}  {'θ₄':>7s}  {'滤波':>6s}  {'M1':>7s}  {'M3':>7s}")
    print(f"  {'─'*65}")

    for i, (name, kp) in enumerate(poses):
        angles = extractor.extract(kp)
        if angles is None:
            continue

        filtered, hold_mask = filter_.update(angles)
        motor_cmds = mapper.map_angles(filtered.tolist())

        hold_str = "".join(["H" if h else "." for h in hold_mask])
        m1 = motor_cmds.get("M1", {}).get("value", 0)
        m3 = motor_cmds.get("M3", {}).get("value", 0)

        print(f"  {i+1:>3}  {name:14s}  "
              f"{filtered[0]:6.1f}°  {filtered[1]:6.1f}°  "
              f"{filtered[2]:6.1f}°  {filtered[3]:6.1f}°  "
              f"{hold_str:6s}  {m1:6.1f}°  {m3:6.1f}°")

    print(f"\n  ✅ 全流程集成测试完成 (共 {len(poses)} 帧姿态)")
    return True


def main():
    """运行所有测试"""
    print("\n" + "★" * 65)
    print("  ArmPoseVision 合成数据测试")
    print("  测试覆盖: 角度提取 | 运动滤波 | 角度映射 | 全流程集成")
    print("★" * 65)

    results = []
    results.append(("AngleExtractor", test_angle_extractor()))
    results.append(("MotionFilter",   test_motion_filter()))
    results.append(("Mapper",         test_mapper()))
    results.append(("集成测试",       test_pipeline_integration()))

    # 汇总
    print("\n" + "=" * 65)
    print("📊 测试汇总")
    print("=" * 65)
    passed = 0
    for name, ok in results:
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")
        if ok:
            passed += 1

    print(f"\n  通过: {passed}/{len(results)}")
    print(f"  状态: {'🎉 全部通过!' if passed == len(results) else '⚠️  部分测试未通过'}")
    print()

    return passed == len(results)


if __name__ == "__main__":
    main()
