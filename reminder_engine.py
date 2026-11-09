"""Day 4：提醒引擎 —— 状态机 + 迟滞 + 冷却。只吃 PostureState，不碰摄像头/UI。

状态（docs/细节规格.md 第二节）：
    NORMAL → COUNTDOWN（越线计时）→ ALERTED（到点提醒）→（冷却后重复提醒）
    ALERTED → NORMAL（连续好姿势 ≥ recover_seconds）
    ALERTED → SNOOZED（用户点小睡）→ NORMAL（静默期满重新计数）

防疲劳三件套：迟滞（recover_seconds）+ 冷却（cooldown_seconds）+ 静默/上限（本类只管前两个，
静默时段与每日上限属 UI 层，Day 5 接）。

update() 每帧调用，返回 None 或 action dict：
    {"type": "alert", "level": "warn"|"severe", "text": str, "seconds": float}
    {"type": "silence"}    # 置信度不足，状态机冻结
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from config import REMINDERS
from posture_rules import PostureState

NORMAL, COUNTDOWN, ALERTED, SNOOZED = "normal", "countdown", "alerted", "snoozed"


@dataclass
class ReminderEngine:
    clock: Callable[[], float] = time.monotonic
    trigger_seconds: float = REMINDERS.trigger_seconds
    recover_seconds: float = REMINDERS.recover_seconds
    cooldown_seconds: float = REMINDERS.cooldown_seconds
    escalate_seconds: float = REMINDERS.escalate_seconds
    snooze_seconds: float = REMINDERS.snooze_seconds
    state: str = NORMAL
    bad_since: Optional[float] = None
    good_since: Optional[float] = None
    last_alert_at: Optional[float] = None
    snoozed_until: Optional[float] = None

    def update(self, ps: PostureState) -> Optional[Dict]:
        now = self.clock()
        level = ps.level

        if level == "unknown":
            # 人离开工位/丢失追踪：不仅冻结，还要清掉坏姿势累计，
            # 否则回来后旧计时继续跑，把"不在工位"的时间算成坏姿势持续 → 误提醒
            self.bad_since = None
            if self.state == COUNTDOWN:
                self.state = NORMAL
            return {"type": "silence"}

        if level == "good":
            return self._on_good(now, ps)
        return self._on_bad(now, ps)

    # -- 内部转移 -------------------------------------------------

    def _on_good(self, now: float, ps: PostureState) -> Optional[Dict]:
        self.bad_since = None
        if self.good_since is None:
            self.good_since = now
        if self.state in (COUNTDOWN, ALERTED) and now - self.good_since >= self.recover_seconds:
            self.state = NORMAL
            self.last_alert_at = None
        elif self.state == SNOOZED and self.snoozed_until is not None and now >= self.snoozed_until:
            self.state = NORMAL
            self.snoozed_until = None
        return None

    def _on_bad(self, now: float, ps: PostureState) -> Optional[Dict]:
        self.good_since = None
        if self.bad_since is None:
            self.bad_since = now
        elapsed = now - self.bad_since

        if self.state == SNOOZED:
            if self.snoozed_until is not None and now >= self.snoozed_until:
                self.state = NORMAL
                self.snoozed_until = None
            return None

        if self.state == NORMAL:
            if elapsed >= self.trigger_seconds:
                self.state = ALERTED
                self.last_alert_at = now
                return self._alert(ps, elapsed)
            self.state = COUNTDOWN
            return None

        if self.state == COUNTDOWN:
            if elapsed >= self.trigger_seconds:
                self.state = ALERTED
                self.last_alert_at = now
                return self._alert(ps, elapsed)
            return None

        # ALERTED：冷却后重复提醒（恢复路径已在 _on_good 处理）
        if self.last_alert_at is not None and now - self.last_alert_at >= self.cooldown_seconds:
            self.last_alert_at = now
            return self._alert(ps, elapsed)
        return None

    def _alert(self, ps: PostureState, elapsed: float) -> Dict:
        severe = elapsed >= self.escalate_seconds
        return {
            "type": "alert",
            "level": "severe" if severe else "warn",
            "text": self._message(ps, elapsed, severe),
            "seconds": round(elapsed, 1),
        }

    @staticmethod
    def _message(ps: PostureState, elapsed: float, severe: bool) -> str:
        m, s = divmod(int(elapsed), 60)
        dur = f"{m} 分 {s} 秒" if m else f"{s} 秒"
        trigger = " / ".join(ps.reasons) if ps.reasons else ps.text
        if severe:
            return f"已经连续姿势不良 {dur}（{trigger}），建议先停下休息"
        return f"姿势不良 {dur}（{trigger}），调整一下"

    # -- 用户动作 -------------------------------------------------

    def snooze(self) -> None:
        self.state = SNOOZED
        self.snoozed_until = self.clock() + self.snooze_seconds

    def reset(self) -> None:
        self.state = NORMAL
        self.bad_since = self.good_since = None
        self.last_alert_at = self.snoozed_until = None
