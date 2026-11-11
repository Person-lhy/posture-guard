"""Day 9（M2-2）：久坐计时器 —— 坐满阈值提醒起身，离座自动清零。

与姿态提醒分开：姿态看"姿势对不对"，这里看"坐了多久"，两条提醒线独立。
冷却期内不再重复提醒（防止每帧轰炸）；冷却结束后仍未起身则再提醒一次。

用法：worker 每帧调 update(level)，返回 None 或 {"type": "sedentary", ...}。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


@dataclass
class SedentaryTracker:
    threshold_seconds: float = 45 * 60.0   # 坐多久提醒
    cooldown_seconds: float = 15 * 60.0    # 两次提醒的最小间隔
    clock: Callable[[], float] = time.monotonic

    sitting_since: Optional[float] = None
    last_alert_at: Optional[float] = None

    def update(self, level: str) -> Optional[Dict]:
        """level 取 PostureState.level；unknown = 人不在 → 视为离座。"""
        now = self.clock()

        if level == "unknown":
            self.sitting_since = None       # 离座清零
            self.last_alert_at = None
            return None

        if self.sitting_since is None:
            self.sitting_since = now        # 刚坐下
            return None

        elapsed = now - self.sitting_since
        if elapsed < self.threshold_seconds:
            return None

        # 坐满阈值：冷却中只记录不提醒
        if self.last_alert_at is not None and now - self.last_alert_at < self.cooldown_seconds:
            return None

        self.last_alert_at = now
        m, s = divmod(int(elapsed), 60)
        dur = f"{m} 分 {s} 秒" if m else f"{s} 秒"
        return {
            "type": "sedentary",
            "text": f"已经连续坐了 {dur}，起来走 2 分钟吧",
            "seconds": round(elapsed, 1),
        }

    def sitting_seconds(self) -> float:
        if self.sitting_since is None:
            return 0.0
        return self.clock() - self.sitting_since

    def reset(self) -> None:
        self.sitting_since = None
        self.last_alert_at = None
