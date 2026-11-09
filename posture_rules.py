"""Day 4：滤波 + 分级 + 迟滞判定 —— 纯函数/可测状态类，无 I/O。

职责（对应 docs/细节规格.md 第二节）：
- MedianFilter：滑动中值滤波（抗野值，不用均值）。
- Grade：单指标迟滞分级（进线/出线不同阈值，防横跳），有状态、可测。
- analyze()：裸值 dict（metrics.py）→ 滤波 → 逐指标分级 → 合成 PostureState。

判定与提醒分离：本文件只回答"现在姿势好不好"；何时发声在 reminder_engine.py。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Deque, Dict, List, Optional

from config import THRESHOLDS

Level = str  # "good" | "warn" | "bad" | "unknown"
_LEVEL_RANK = {"good": 0, "warn": 1, "bad": 2}


class MedianFilter:
    """滑动中值滤波。样本不足窗口一半时返回 None（数据太少，不算）。"""

    def __init__(self, window: int = THRESHOLDS.median_window):
        self.window = window
        self._buf: Dict[str, Deque[float]] = {}

    def push(self, key: str, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        buf = self._buf.setdefault(key, deque(maxlen=self.window))
        buf.append(float(value))
        if len(buf) < max(1, self.window // 2 + 1):
            return None
        return float(median(buf))

    def reset(self) -> None:
        self._buf.clear()


class Grade:
    """单指标迟滞分级。

    进线（warn→warn）用 warn 阈值；回线（warn→good）必须回到 recover 阈值内，
    两线之间维持原级 —— 这就是防抖动的迟滞。持续 ≥ severe 判 bad。
    """

    def __init__(self, warn: float, recover: float, severe: Optional[float] = None):
        assert recover <= warn, "recover 阈值必须 ≤ warn（出线比进线松）"
        self.warn = warn
        self.recover = recover
        self.severe = severe
        self.level: Level = "good"

    def update(self, value: Optional[float]) -> Level:
        if value is None:
            # 数据缺失：good→unknown（如实报告）；warn/bad 冻结（不因丢帧误恢复）
            return "unknown" if self.level == "good" else self.level
        v = abs(value)
        if self.level == "good":
            if self.severe is not None and v >= self.severe:
                self.level = "bad"     # 直接跳到严重（如久坐后猛低头）
            elif v >= self.warn:
                self.level = "warn"
        elif self.level == "warn":
            if self.severe is not None and v >= self.severe:
                self.level = "bad"
            elif v < self.recover:
                self.level = "good"
        else:  # bad
            if v < self.recover:
                self.level = "good"    # 跌破出线 → 恢复
            elif self.severe is None or v < self.severe:
                self.level = "warn"    # 退出严重区间 → 降为 warn
        return self.level

    def reset(self) -> None:
        self.level = "good"


@dataclass
class PostureState:
    level: Level                          # 合成级 = 未知道数
    details: Dict[str, Level]             # 逐指标分级
    reasons: List[str]                    # 越线项，如 ["theta +28.3 (warn)"]
    text: str = ""                        # 人读状态串，如 "WARN theta +28.3 drop +0.42"


def _fmt(key: str, value: Optional[float], unit: str = "") -> str:
    if value is None:
        return f"{key} n/a"
    return f"{key} {value:+.2f}{unit}"


def analyze(raw: Dict[str, Optional[float]], mf: MedianFilter,
            grades: Dict[str, Grade]) -> PostureState:
    """裸值 → 滤波 → 分级 → 合成状态。

    raw：metrics.extract_metrics 的输出。
    mf / grades：调用方持有（跨帧状态），grades 至少含
        theta_deg / head_drop / head_pitch_deg / shoulder_tilt 四项。
    """
    # 追踪丢失（raw 全空 = 人不在画面）：清掉滤波器旧缓冲，否则离开工位后
    # 中值仍被旧值喂着报 warn，状态机收不到 unknown，会把"不在工位"算成坏姿势。
    # 注意：只在 raw 全空时 reset，不能用 f 全空判断——预热期（样本不足）f 也是 None
    if all(raw.get(k) is None for k in ("theta_deg", "head_drop", "head_pitch_deg",
                                        "neck_ratio", "shoulder_tilt")):
        mf.reset()

    f = {k: mf.push(k, raw.get(k)) for k in
         ("theta_deg", "head_drop", "head_pitch_deg", "neck_ratio", "shoulder_tilt")}

    details: Dict[str, Level] = {
        "theta_deg": grades["theta_deg"].update(f["theta_deg"]),
        "head_drop": grades["head_drop"].update(f["head_drop"]),
        "head_pitch_deg": grades["head_pitch_deg"].update(f["head_pitch_deg"]),
        "shoulder_tilt": grades["shoulder_tilt"].update(f["shoulder_tilt"]),
    }
    # r 颈长比需个人基线，M3 才判级；先只报告数值
    neck = f["neck_ratio"]
    details["neck_ratio"] = "unknown" if neck is None else "good"  # good=已测未判级

    # 合成：取未知道数；一票否决 warn/bad。
    # 追踪丢失（全部滤波值为空）时强制 unknown：单指标冻结（局部遮挡）仍成立，
    # 但合成级必须如实报"人不在"，否则状态机收不到信号，把离开工位算成坏姿势
    if all(v is None for v in f.values()):
        level = "unknown"
    elif all(v == "unknown" for v in details.values()):
        level = "unknown"
    else:
        level = "good"
        for v in details.values():
            if _LEVEL_RANK.get(v, -1) > _LEVEL_RANK[level]:
                level = v

    reasons = [f"{k} {v}" for k, v in details.items() if v in ("warn", "bad")]
    parts = [
        _fmt("theta", f["theta_deg"], "deg"),
        _fmt("drop", f["head_drop"]),
        _fmt("pitch", f["head_pitch_deg"], "deg"),
        _fmt("shldr", f["shoulder_tilt"]),
    ]
    text = f"{level.upper()} " + " ".join(parts)

    return PostureState(level=level, details=details, reasons=reasons, text=text)


def default_grades() -> Dict[str, Grade]:
    """按 config 阈值建一套分级器（迟滞：出线阈值 < 进线阈值）。"""
    t = THRESHOLDS
    return {
        "theta_deg": Grade(t.neck_tilt_warn, t.neck_tilt_recover, t.neck_tilt_severe),
        "head_drop": Grade(t.head_drop_warn, t.head_drop_recover),
        "head_pitch_deg": Grade(t.neck_tilt_warn, t.neck_tilt_recover, t.neck_tilt_severe),
        "shoulder_tilt": Grade(t.shoulder_tilt_warn, t.shoulder_tilt_recover),
    }
