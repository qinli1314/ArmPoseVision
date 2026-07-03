# ArmPoseVision 手臂姿态视觉识别系统

基于单目摄像头的人体手臂姿态实时识别与角度分析系统。

## 功能概述

通过摄像头实时捕捉人体手臂动作，提取**肩/肘/腕 6 个自由度角度**，实时显示在 GUI 界面上。

| 角度 | 名称 | 动作 |
|------|------|------|
| θ₁ | 肩上下 | 抬臂/侧平举/上举 |
| θ₂ | 肩左右 | 手臂前后摆动 |
| θ₃ | 肘屈伸 | 弯肘/伸肘 |
| θ₄ | 前臂旋转 | 手心朝上/朝下 |
| θ₅ | 腕 Pitch | 手腕上下弯曲 |
| θ₆ | 腕 Roll | 手腕左右旋转 |

## 📸 界面预览

```
┌──────────────────────────────────────────────────┐
│ 📷 [Camera 0 ▼]  [▶ 开始] [■ 停止] [🦾右臂▼] [📱前置▼] │
├──────────────────────────┬───────────────────────┤
│                          │  📐 关节角度          │
│    视频画面 + 骨架叠加    │  θ₁ 肩上下 ████░░ 45°  │
│                          │  θ₂ 肩左右 ██░░░░ 12°  │
│    (右上角实时角度数值)   │  θ₃ 肘屈伸 ██████ 89°  │
│                          │  θ₄ 前臂旋 ██░░░░ -15° │
│                          │  θ₅ 腕Pitch ███░░░ 5°  │
│                          │  θ₆ 腕Roll  ██░░░░ -3° │
├──────────────────────────┴───────────────────────┤
│ [12:00:01] 摄像头已连接                          │
│ [12:00:02] 开始采集，FPS: 29.8                   │
└──────────────────────────────────────────────────┘
```

##  快速开始

### 环境要求

- Windows 10/11
- Python 3.10
- USB 摄像头（可选，视频文件也可测试）

### 安装

```bash
# 1. 创建虚拟环境
conda create -n armpose2robot python=3.10 -y
conda activate armpose2robot

# 2. 安装依赖
cd ArmPose2Robot
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 运行

```bash
# USB 摄像头模式（默认前置）
python main.py

# 后置摄像头
python main.py --rear

# 视频文件回放
python main.py --video test_videos/肩前后.mp4

# 视频文件 + 后置模式
python main.py --video test_videos/肩前后.mp4 --rear
```

### 测试（无需摄像头）

```bash
python test/synthetic_test.py
```

##  操作说明

### 工具栏

| 按钮 | 功能 |
|------|------|
|  **摄像头下拉框** | 选择 USB 摄像头设备 |
| **开始采集** | 打开摄像头/开始视频回放 |
| **停止** | 停止采集 |
| **右臂/左臂** | 切换追踪手臂（运行时可用） |
| **前置/后置** | 切换镜像模式（运行时可用） |
| **重载映射** | 热重载映射配置文件 |

### 角度面板

- 每个角度带**进度条**，直观显示角度位置
- 绿色 = 正常，黄色 = 保持中（滤波触发）
- 底部显示检测状态

### 视频画面

- 实时**骨架叠加**（绿色=躯干，蓝色=手臂关节点）
- 手部检测启用时显示**黄色手部关键点**
- 右上角**实时角度数值**叠加

##  配置文件

### `config/app_config.json` — 应用配置

```json
{
  "camera": { "resolution": [1280, 720], "target_fps": 30 },
  "filter": { "dead_zone_deg": 5.0, "ema_alpha": 0.3, "hold_timeout_ms": 500 },
  "pose": { "arm": "right", "detection_confidence": 0.5 }
}
```

> 修改后自动热重载，无需重启程序。

### `config/mapping_config.json` — 角度→电机映射

```json
{
  "mappings": [
    {"angle_index": 1, "motor_id": "M1", "scale": 1.0, "offset": 0.0, "invert": false},
    ...
  ],
  "angle_limits": {
    "M1": {"min": 0, "max": 180},
    ...
  }
}
```

## 打包为 exe

```bash
python build.py
```

打包后生成 `dist/ArmPoseVision/ArmPoseVision.exe`，客户双击即可运行。

## 模块架构

```
摄像头/视频文件 ─→ 图像采集 (camera.py)
                       │
                       ▼
                  姿态检测 (pose_detector.py) ─→ MediaPipe Pose + Hands
                       │
                       ▼
                  6DOF 角度计算 (angle_extractor.py)
                       │
                       ▼
                  运动滤波 (motion_filter.py) ─→ Dead Zone + EMA
                       │
                       ▼
                  角度映射 (mapper.py) ─→ 配置驱动
                       │
                       ▼
                  GUI 显示 (main_window.py)
                       ├── 视频画面 + 骨架叠加
                       ├── 角度面板 + 进度条
                       ├── 日志输出
                       └── 角度数据文件记录
```

## 项目结构

```
ArmPose2Robot/
├── main.py                    # 程序入口
├── build.py                   # 打包脚本
├── requirements.txt           # 依赖
├── config/                    # 配置文件
│   ├── app_config.json
│   └── mapping_config.json
├── vision/                    # 视觉模块
│   ├── camera.py              # 摄像头/视频采集
│   ├── pose_detector.py       # 姿态+手部检测
│   └── angle_extractor.py     # 6DOF角度计算
├── filter/
│   └── motion_filter.py       # 运动滤波
├── mapping/
│   └── mapper.py              # 角度映射
├── network/
│   └── tcp_client.py          # TCP通信（预留）
├── gui/
│   ├── main_window.py         # 主窗口
│   ├── video_widget.py        # 视频显示
│   └── angle_panel.py         # 角度面板
├── test/
│   ├── synthetic_test.py      # 合成数据测试
│   └── scan_camera.py         # 摄像头扫描
├── logs/                      # 运行日志
└── test_videos/               # 测试视频
```

## 角度计算原理

角度从 MediaPipe 提取的 33 个人体关键点中计算：

- **θ₁ 肩上下**：大臂向量与躯干向量的夹角 → 0°=下垂, 90°=平举, 180°=上举
- **θ₂ 肩左右**：大臂在水平面投影与侧向轴的夹角 → 0°=身侧, 90°=前伸
- **θ₃ 肘屈伸**：大臂与小臂向量的夹角 → 0°=伸直, 90°=直角
- **θ₄ 前臂旋转**：肘关节平面法向量绕小臂轴的旋转
- **θ₅ 腕Pitch**：基于深度信息估算 + 手部关键点增强
- **θ₆ 腕Roll**：小臂横向偏移 + 手部关键点增强

## 日志

角度数据自动写入 `logs/angle_log_YYYYMMDD.txt`：

```json
{"timestamp": "2026-07-03T14:32:05.123", "angles": {"theta_1": 45.3, "theta_2": 12.7, ...}, "filtered": false}
```

## 后续方向
- 机械臂网络通信（代码已预留）


