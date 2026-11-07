"""可调参数集中于此 —— 全部为 M1 起步值，M2/M3 用实测数据校准。

原则：判定阈值进 Thresholds，提醒节奏进 Reminders，采集参数进 Camera。
业务代码不要写魔法数字。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Thresholds:
    # θ 颈线偏角（度）
    neck_tilt_warn: float = 25.0
    neck_tilt_severe: float = 40.0
    neck_tilt_recover: float = 18.0
    # d 低头比值（无量纲，鼻下移量 / 肩宽）
    head_drop_warn: float = 0.35
    head_drop_recover: float = 0.28
    # s 双肩水平差（无量纲）
    shoulder_tilt_warn: float = 0.15
    shoulder_tilt_recover: float = 0.10
    # r 颈长比低于个人基线的比例（M3 校准）
    neck_length_drop: float = 0.85
    # 数据质量
    confidence_min: float = 0.7
    median_window: int = 5


@dataclass(frozen=True)
class Reminders:
    trigger_seconds: float = 15.0        # 坏姿势持续多久才响
    recover_seconds: float = 5.0         # 连续好姿势多久算恢复（迟滞）
    cooldown_seconds: float = 300.0      # 两次提醒的最小间隔
    escalate_seconds: float = 600.0      # 同一次坏姿势持续多久升级为"严重"
    sedentary_seconds: float = 45 * 60.0  # 久坐提醒
    snooze_seconds: float = 30 * 60.0    # 用户点"小睡"的静默时长
    quiet_hours: Tuple[str, str] = ("23:00", "08:00")  # 静默时段
    daily_alert_cap: int = 20            # 每日提醒上限，超限只变色不弹窗


@dataclass(frozen=True)
class Camera:
    index: int = 0
    width: int = 1280
    height: int = 720
    process_width: int = 640  # 推理输入宽度，降载用


THRESHOLDS = Thresholds()
REMINDERS = Reminders()
CAMERA = Camera()
