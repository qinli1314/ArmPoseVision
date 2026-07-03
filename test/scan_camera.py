"""Scan cameras on system"""
import cv2

print("扫描摄像头...")
found = False
for api in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i, api)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    name = cap.getBackendName()
                    print(f"  ✅ Camera {i}: {w}x{h}  (后端: {name})")
                    found = True
                cap.release()
        except:
            pass

if not found:
    print("  ❌ 未检测到摄像头")
    print("\n可能的原因：")
    print("  1. 电脑没有摄像头或驱动未安装")
    print("  2. 摄像头被其他程序占用")
    print("  3. 检查 设备管理器 → 相机 确认是否有摄像头")
print("\n扫描完成")
