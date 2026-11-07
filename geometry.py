"""几何与角度计算 —— 纯函数、无 I/O、无第三方依赖，可单测。

坐标系约定
----------
- 图像像素系：原点左上，x 向右，**y 向下**（图像惯例，与数学坐标系相反）。
- MediaPipe 归一化 landmark：x/y 属于 [0,1]，z 与 x 同尺度（相对髋中点，靠近摄像头为负）。
- 换算到像素：px = x*W, py = y*H, pz = z*W。
- 竖直向上方向 = (0, -1)（因为 y 轴向下）。

约定：所有计算函数入参为像素坐标元组；返回 None 表示输入退化（零向量/零尺度），
调用方应跳过本帧，不要把 None 当 0 使用。
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

Point = Tuple[float, float]
Point3 = Tuple[float, float, float]


def to_pixels(lm: Any, width: int, height: int) -> Point3:
    """MediaPipe landmark（.x/.y/.z 归一化）→ 像素坐标 (px, py, pz)。

    z 与 x 同尺度，故乘 width 而不是 height（MediaPipe 约定）。
    """
    return (lm.x * width, lm.y * height, lm.z * width)


def midpoint(a: Point, b: Point) -> Point:
    """两点中点。"""
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def neck_tilt_deg(ear_mid: Point, sh_mid: Point) -> Optional[float]:
    """θ：颈线（肩中点→耳中点）相对竖直向上的偏角，单位度。

    v = ear_mid - sh_mid（图像系 y 向下，耳在肩上方，故 v.y < 0）
    θ = atan2(v.x, -v.y)，即 tan θ = v.x / (-v.y)

    正值 = 耳偏图像右侧，负值 = 偏左。正面相机下主要反映侧向偏移/颈线斜率，
    矢状面前移请配合 neck_length_ratio 与 head_pitch_deg（z 辅助）。
    """
    vx = ear_mid[0] - sh_mid[0]
    vy = ear_mid[1] - sh_mid[1]
    if math.hypot(vx, vy) < 1e-6:
        return None
    return math.degrees(math.atan2(vx, -vy))


def head_drop_ratio(nose: Point, ear_mid: Point, scale: float) -> Optional[float]:
    """d：低头主指标（正面相机）。鼻相对耳中点的垂直下移量 / scale。

    d > 0 表示鼻低于耳（低头）。scale 用肩宽等稳定尺度（像素）。
    侧视"鼻-耳与水平夹角"在正面相机会退化（鼻与耳几乎同 x），故用比值而非角度。
    """
    if scale <= 1e-6:
        return None
    return (nose[1] - ear_mid[1]) / scale


def head_pitch_deg(nose: Point3, ear_mid: Point3) -> Optional[float]:
    """φ：俯仰角（实验性，依赖 z）。q = nose - ear_mid，φ = atan2(q.y, |q.z|)。

    正值 = 低头。z 噪声大或 q 接近零向量时返回 None，只作辅助指标。
    """
    qy = nose[1] - ear_mid[1]
    qz = nose[2] - ear_mid[2]
    if math.hypot(qy, qz) < 1e-6:
        return None
    return math.degrees(math.atan2(qy, abs(qz)))


def neck_length_ratio(ear_mid: Point, sh_mid: Point, scale: float) -> Optional[float]:
    """r：颈长比（前移/含胸代理）。|ear_mid - sh_mid| / scale。

    正面相机下头前移会让耳-肩距离在图像里缩短，r 变小约等于含胸前倾。
    个体与机位差异大，建议相对个人基线判级（M3 自适应阈值）。
    """
    if scale <= 1e-6:
        return None
    return math.hypot(ear_mid[0] - sh_mid[0], ear_mid[1] - sh_mid[1]) / scale


def shoulder_tilt(sh_l: Point, sh_r: Point) -> Optional[float]:
    """s：双肩水平差 |Δy| / |Δx|（无量纲）。正面视角下最稳的指标之一。"""
    dx = sh_l[0] - sh_r[0]
    dy = sh_l[1] - sh_r[1]
    if abs(dx) < 1e-6:
        return None
    return abs(dy / dx)
