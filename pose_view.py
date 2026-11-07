"""Day 2+3：MediaPipe 骨架叠加 + 实时角度指标。

摄像头取流 → MediaPipe Pose 33 关键点 → 骨架/锚点绘制 → 角度指标 HUD。

用法：
    python pose_view.py         # 默认摄像头 0
    python pose_view.py 1       # 指定设备号

按键：q 退出。
- FPS 显示 instant（瞬时）与 avg（平滑，看这个）。
- 右上角 HUD 显示 θ / d / r / φ / s，颜色随阈值变化（绿=正常 黄=警告 红=严重）。
- 终端每 1s 打印一行裸值（Day 3 调试用；滤波/分级在 Day 4）。
"""

from __future__ import annotations

import sys
import time

import cv2
import mediapipe as mp

from config import THRESHOLDS
from metrics import ANCHORS, extract_metrics

# 锚点索引（MediaPipe Pose 33 点约定），与 metrics.py 保持一致
NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

# 骨架线色（BGR）与关键锚点色
SKELETON_COLOR = (230, 220, 90)      # 青蓝
ANCHOR_COLOR = (60, 200, 255)        # 橙黄
ANCHOR_RADIUS = 6
ANCHOR_LABELS = {NOSE: "nose", LEFT_EAR: "earL", RIGHT_EAR: "earR",
                 LEFT_SHOULDER: "shL", RIGHT_SHOULDER: "shR"}

# 状态色（BGR）：good / warn / bad / unknown
STATUS_COLORS = {"good": (80, 220, 80), "warn": (60, 200, 255),
                 "bad": (60, 60, 230), "unknown": (170, 170, 170)}

PRINT_INTERVAL = 1.0   # 终端节流：1s 一行


def _level(value, warn, severe):
    """按阈值分级：None→unknown，|v|≥severe→bad，≥warn→warn，否则 good。"""
    if value is None:
        return "unknown"
    v = abs(value)
    if v >= severe:
        return "bad"
    if v >= warn:
        return "warn"
    return "good"


def _level_pair(value, warn, recover):
    """双阈值分级（进/出线不同，防抖动）：越 warn→warn，越 recover→good，之间→unknown。

    单帧无状态，这里只做静态分级；迟滞状态机在 Day 4 的 posture_rules.py。
    """
    if value is None:
        return "unknown"
    v = abs(value)
    if v >= warn:
        return "warn"
    if v <= recover:
        return "good"
    return "unknown"


def main() -> int:
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"无法打开摄像头 {idx}。检查：1) 是否被其他软件占用  2) 设备号是否正确")
        return 1

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    # model_complexity=1：精度/速度平衡（0 更快，2 更准）
    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      smooth_landmarks=True,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        window = "posture-guard pose (q to quit)"
        fps_avg = 0.0
        last = time.perf_counter()
        last_print = 0.0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("取流失败，摄像头被拔出或被占用")
                    return 1

                now = time.perf_counter()
                dt = now - last
                last = now
                fps_inst = 1.0 / dt if dt > 0 else 0.0
                fps_avg = fps_inst if fps_avg == 0.0 else 0.9 * fps_avg + 0.1 * fps_inst

                # MediaPipe 要 RGB；BGR → RGB → 推理 → BGR 显示
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False   # 提速
                result = pose.process(frame_rgb)

                h, w = frame.shape[:2]
                if result.pose_landmarks:
                    lms = result.pose_landmarks.landmark
                    # 1) 整副骨架
                    mp_draw.draw_landmarks(
                        frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_draw.DrawingSpec(
                            color=SKELETON_COLOR, thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_draw.DrawingSpec(
                            color=SKELETON_COLOR, thickness=2, circle_radius=2))
                    # 2) Day 3 锚点高亮
                    for i in ANCHORS:
                        lm = lms[i]
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), ANCHOR_RADIUS, ANCHOR_COLOR, -1)
                        cv2.putText(frame, ANCHOR_LABELS[i], (cx + 8, cy - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ANCHOR_COLOR, 1,
                                    cv2.LINE_AA)
                    # 3) Day 3 指标：提取 → HUD → 终端节流打印
                    m = extract_metrics(lms, w, h)
                    _draw_hud(frame, m, fps_inst, fps_avg)
                    if now - last_print >= PRINT_INTERVAL:
                        last_print = now
                        _print_metrics(m, fps_avg)
                else:
                    cv2.putText(frame, "no person detected", (12, 48),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 230), 2,
                                cv2.LINE_AA)
                    cv2.putText(frame, f"FPS instant: {fps_inst:4.1f}  avg: {fps_avg:4.1f}",
                                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 220, 80), 2,
                                cv2.LINE_AA)

                cv2.imshow(window, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
    return 0


def _draw_hud(frame, m: dict, fps_inst: float, fps_avg: float) -> None:
    """右上角 HUD：FPS + 五项指标，颜色随阈值变化。"""
    h, w = frame.shape[:2]
    lines = [
        (f"FPS avg {fps_avg:4.1f}", "good"),
        (f"theta {m['theta_deg']:+6.1f} deg",
         _level(m["theta_deg"], THRESHOLDS.neck_tilt_warn, THRESHOLDS.neck_tilt_severe)),
        (f"drop  {m['head_drop']:+6.2f}",
         _level_pair(m["head_drop"], THRESHOLDS.head_drop_warn, THRESHOLDS.head_drop_recover)),
        (f"neck  {m['neck_ratio']:6.2f}", "good"),
        (f"pitch {m['head_pitch_deg']:+6.1f} deg",
         _level(m["head_pitch_deg"], THRESHOLDS.neck_tilt_warn, THRESHOLDS.neck_tilt_severe)),
        (f"shldr {m['shoulder_tilt']:6.2f}",
         _level_pair(m["shoulder_tilt"], THRESHOLDS.shoulder_tilt_warn,
                     THRESHOLDS.shoulder_tilt_recover)),
        (f"conf  {m['confidence']:6.2f}",
         "good" if m["confidence"] >= THRESHOLDS.confidence_min else "unknown"),
    ]
    x = w - 250
    y = 30
    cv2.rectangle(frame, (x - 12, 12), (w - 12, 26 + 26 * len(lines)),
                  (30, 30, 30), -1)
    for text, level in lines:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    STATUS_COLORS[level], 2, cv2.LINE_AA)
        y += 26


def _print_metrics(m: dict, fps_avg: float) -> None:
    """终端 1s 一行：裸值（未滤波），Day 3 调试用。"""
    def fmt(v, spec="+6.1f"):
        return format(v, spec) if v is not None else "   n/a"

    print(f"fps {fps_avg:4.1f} | theta {fmt(m['theta_deg'])} | drop {fmt(m['head_drop'], '+6.2f')}"
          f" | neck {fmt(m['neck_ratio'], '6.2f')} | pitch {fmt(m['head_pitch_deg'])}"
          f" | shldr {fmt(m['shoulder_tilt'], '6.2f')} | conf {m['confidence']:5.2f}")


if __name__ == "__main__":
    sys.exit(main())
