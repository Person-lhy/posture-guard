"""从 MediaPipe 33 关键点提取坐姿指标 —— 纯函数，可单测。

流水线：landmark（归一化）→ 像素坐标 → 几何指标（geometry.py）→ 裸值 dict。
滤波 / 分级 / 迟滞判定属 Day 4（posture_rules.py），本文件不算。

约定（与 geometry.py / docs/细节规格.md 一致）：
- 图像像素系 y 向下；MediaPipe z 乘 width。
- 任一参与关键点 visibility < 阈值 → 该指标返回 None（调用方跳过，勿当 0）。
- scale = 双肩像素距，作无量纲指标（d / r）的分母。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from config import THRESHOLDS
from geometry import (
    head_drop_ratio,
    head_pitch_deg,
    midpoint,
    neck_length_ratio,
    neck_tilt_deg,
    shoulder_tilt,
    to_pixels,
)

# MediaPipe Pose 33 点约定
NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

ANCHOR_INDICES = (NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER)


def extract_metrics(landmarks: Sequence[Any], width: int, height: int) -> Dict[str, Optional[float]]:
    """landmarks：MediaPipe landmark 序列（.x/.y/.z/.visibility，归一化）。

    返回裸值 dict（未滤波）：
        theta_deg   θ 颈线偏角（度，带符号）
        head_drop   d 低头比值（无量纲）
        head_pitch_deg φ 俯仰角（度，依赖 z，实验性）
        neck_ratio  r 颈长比（无量纲）
        shoulder_tilt s 双肩水平差（无量纲）
        scale       双肩像素距（调试/日志用）
        confidence  锚点最低可见度（数据质量）
    任何一项算不出 → 该值为 None。
    """
    conf = min(float(landmarks[i].visibility) for i in ANCHOR_INDICES)
    out: Dict[str, Optional[float]] = {
        "theta_deg": None,
        "head_drop": None,
        "head_pitch_deg": None,
        "neck_ratio": None,
        "shoulder_tilt": None,
        "scale": None,
        "confidence": conf,
    }
    if conf < THRESHOLDS.confidence_min:
        return out

    nose = to_pixels(landmarks[NOSE], width, height)
    ear_l = to_pixels(landmarks[LEFT_EAR], width, height)
    ear_r = to_pixels(landmarks[RIGHT_EAR], width, height)
    sh_l = to_pixels(landmarks[LEFT_SHOULDER], width, height)
    sh_r = to_pixels(landmarks[RIGHT_SHOULDER], width, height)

    ear_mid = midpoint(ear_l[:2], ear_r[:2])
    ear_mid_z = (ear_l[2] + ear_r[2]) / 2.0
    sh_mid = midpoint(sh_l[:2], sh_r[:2])
    scale = ((sh_l[0] - sh_r[0]) ** 2 + (sh_l[1] - sh_r[1]) ** 2) ** 0.5

    out["theta_deg"] = neck_tilt_deg(ear_mid, sh_mid)
    out["head_drop"] = head_drop_ratio(nose[:2], ear_mid, scale)
    out["head_pitch_deg"] = head_pitch_deg(nose, (ear_mid[0], ear_mid[1], ear_mid_z))
    out["neck_ratio"] = neck_length_ratio(ear_mid, sh_mid, scale)
    out["shoulder_tilt"] = shoulder_tilt(sh_l[:2], sh_r[:2])
    out["scale"] = scale if scale > 1e-6 else None
    return out
