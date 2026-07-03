"""
打包脚本 — 将 ArmPoseVision 打包为独立 exe
用法:
    conda activate armpose2robot
    python build.py

打包后的 exe 在 dist/ArmPoseVision/ 目录下
"""

import os
import sys
import shutil
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(BASE_DIR, "dist")


def main():
    print("=" * 60)
    print("  ArmPoseVision 打包工具")
    print("=" * 60)

    # 确保 pyinstaller 已安装
    try:
        import PyInstaller
    except ImportError:
        print("正在安装 pyinstaller...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pyinstaller>=6.0.0",
             "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]
        )

    # 清理旧的构建文件
    for d in ["build", "dist"]:
        path = os.path.join(BASE_DIR, d)
        if os.path.exists(path):
            print(f"清理 {d}/...")
            shutil.rmtree(path, ignore_errors=True)

    # 确保打包时包含配置文件
    config_dir = os.path.join(BASE_DIR, "config")
    if not os.path.exists(config_dir):
        print("❌ 配置文件目录不存在!")
        return

    print("\n开始打包...\n")

    # PyInstaller 命令
    cmd = [
        "pyinstaller",
        "--name", "ArmPoseVision",
        "--windowed",              # 不显示控制台
        "--onefile",               # 单文件
        "--icon", "NONE",          # 无图标
        "--add-data", f"config{os.pathsep}config",
        "--add-data", f"gui{os.pathsep}gui",
        "--add-data", f"vision{os.pathsep}vision",
        "--add-data", f"filter{os.pathsep}filter",
        "--add-data", f"mapping{os.pathsep}mapping",
        "--add-data", f"network{os.pathsep}network",
        "--hidden-import", "PyQt5.sip",
        "--collect-submodules", "mediapipe",
        "--collect-data", "mediapipe",
        "main.py",
    ]

    result = subprocess.run(cmd, cwd=BASE_DIR)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("  ✅ 打包成功!")
        print(f"  📁 输出目录: {os.path.join(DIST_DIR, 'ArmPoseVision')}")
        print(f"  🚀 可执行文件: {os.path.join(DIST_DIR, 'ArmPoseVision', 'ArmPoseVision.exe')}")
        print("\n  注意: 首次启动可能较慢，因为需要加载模型文件")
        print("=" * 60)
    else:
        print(f"\n❌ 打包失败 (返回码: {result.returncode})")
        print("   尝试用 singlefile 模式重新打包...")


if __name__ == "__main__":
    main()
