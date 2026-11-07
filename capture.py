"""Day 1 摄像头取流脚本：验证摄像头可用 + 实时 FPS + 画面显示。

用法：
    python capture.py          # 默认摄像头 0
    python capture.py 1        # 指定设备号

按键：q 退出。
"""

from __future__ import annotations

import sys
import time

import cv2


def main() -> int:
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"无法打开摄像头 {idx}。检查：1) 是否被其他软件占用  2) 设备号是否正确")
        return 1

    # 设定采集分辨率（驱动不支持时会自动回退，无妨）
    from config import CAMERA
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA.height)

    window = "posture-guard capture (q to quit)"
    last = time.perf_counter()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("取流失败，摄像头被拔出或被占用")
                return 1

            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            text = f"FPS: {fps:4.1f}  size: {frame.shape[1]}x{frame.shape[0]}"
            cv2.putText(frame, text, (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(window, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
