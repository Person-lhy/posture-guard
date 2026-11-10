"""Day 7：端到端集成测试 —— 合成关键点流走完整链路。

链路：FakeLM(33点) → metrics.extract_metrics → posture_rules.analyze(滤波+分级)
      → reminder_engine.ReminderEngine(状态机) → storage.append_row(CSV)

验证的是**系统行为**，不是单元行为：一段"低头 2 分钟"该恰好弹几次提醒、
"起身"不该被算进坏姿势、CSV 里各等级行数对不对。
运行：python tests/test_pipeline.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import REMINDERS, THRESHOLDS  # noqa: E402
from metrics import extract_metrics  # noqa: E402
from posture_rules import MedianFilter, analyze, default_grades  # noqa: E402
from reminder_engine import ReminderEngine  # noqa: E402
import storage  # noqa: E402
from tests.test_metrics import FakeLM, _lms  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 0.0
    def __call__(self):
        return self.t
    def tick(self, dt):
        self.t += dt


def _frame(pose: str, w: int = 640, h: int = 480):
    """按姿态名造 33 点关键点（像素系 y 向下）。"""
    if pose == "none":
        return None
    if pose == "good":      # 耳(320,200) 肩(320,300) 鼻(320,228) 肩宽 200
        return _lms(FakeLM(0.5, 0.475, 50.0), FakeLM(0.4375, 5 / 12, 100.0),
                    FakeLM(0.5625, 5 / 12, 100.0),
                    FakeLM(0.34375, 0.625, 100.0), FakeLM(0.65625, 0.625, 100.0))
    if pose == "drop":      # 低头：鼻远低于耳 → d≈0.40
        return _lms(FakeLM(0.5, 280 / 480, 50.0), FakeLM(0.4375, 5 / 12, 100.0),
                    FakeLM(0.5625, 5 / 12, 100.0),
                    FakeLM(0.34375, 0.625, 100.0), FakeLM(0.65625, 0.625, 100.0))
    raise ValueError(pose)


def _run_session(frames, w=640, h=480):
    """跑一整段会话，返回 (alerts, states, rows)。每帧推进假时钟 1 秒。"""
    clock = FakeClock()
    engine = ReminderEngine(clock=clock,
                            trigger_seconds=REMINDERS.trigger_seconds,
                            recover_seconds=REMINDERS.recover_seconds,
                            cooldown_seconds=REMINDERS.cooldown_seconds,
                            escalate_seconds=REMINDERS.escalate_seconds)
    mf = MedianFilter(THRESHOLDS.median_window)
    grades = default_grades()
    alerts, states, rows = [], [], []

    for lm33 in frames:
        if lm33 is None:
            raw = {"theta_deg": None, "head_drop": None, "head_pitch_deg": None,
                   "neck_ratio": None, "shoulder_tilt": None, "confidence": 0.0}
        else:
            raw = extract_metrics(lm33, w, h)
        ps = analyze(raw, mf, grades)
        states.append(ps.level)
        rows.append((ps.level, dict(ps.details), raw.get("confidence", 0.0)))
        action = engine.update(ps)
        if action and action.get("type") == "alert":
            alerts.append(action)
        clock.tick(1.0)
    return alerts, states, rows


def test_full_session_alerts_and_csv():
    """正坐 30s → 低头 120s → 正坐 30s → 起身 20s：
    低头段应恰好 2 次提醒（首响 + 冷却 300s 后重复），起身段 0 次。"""
    d = tempfile.mkdtemp(prefix="pg_pipe_")
    old = storage.DATA_DIR
    storage.DATA_DIR = d
    try:
        frames = [_frame("good")] * 30 + [_frame("drop")] * 120 \
               + [_frame("good")] * 30 + [_frame("none")] * 20
        alerts, states, rows = _run_session(frames)

        # 提醒节奏：15s 触发，冷却 300s；低头段 120s 不到第二个冷却点 → 恰好 1 次
        # （120 < 300，第二次提醒要等 315s，所以是 1 次不是 2 次）
        assert len(alerts) == 1, f"expect 1 alert, got {len(alerts)}: {alerts}"
        assert alerts[0]["type"] == "alert" and alerts[0]["level"] == "warn"
        assert "姿势不良" in alerts[0]["text"]
        assert 14.0 <= alerts[0]["seconds"] <= 16.5   # 15s 触发（滤波有微小延迟）

        # 状态序列：起手滤波预热是 unknown，正坐段 good，低头段 warn/bad，起身 unknown
        assert states[-1] == "unknown"                 # 起身收尾
        assert "good" in states[30:60] and "warn" in states[60:150]

        # CSV 落盘行数 = 帧数
        path = storage.append_row(rows[0][0], rows[0][1], confidence=rows[0][2])
        for level, details, conf in rows[1:]:
            storage.append_row(level, details, confidence=conf)
        import csv as _csv
        with open(path, encoding="utf-8") as f:
            n = sum(1 for _ in _csv.DictReader(f))
        assert n == len(frames)
    finally:
        storage.DATA_DIR = old
        shutil.rmtree(d, ignore_errors=True)


def test_long_slouch_escalates_and_realerts():
    """低头 700s（> 冷却 300s 且 > 升级 600s）：应有 3 次提醒，最后一次为 severe。"""
    frames = [_frame("good")] * 20 + [_frame("drop")] * 700
    alerts, states, _ = _run_session(frames)
    assert len(alerts) == 3, f"expect 3 alerts, got {len(alerts)}"
    assert alerts[0]["level"] == "warn"
    assert alerts[-1]["level"] == "severe"
    assert "停下休息" in alerts[-1]["text"]
    assert states[-1] in ("warn", "bad")


def test_leaving_desk_freezes_not_alert():
    """低头 10s 就起身 60s：坏姿势计时被打断，不该弹提醒。"""
    frames = [_frame("good")] * 10 + [_frame("drop")] * 10 + [_frame("none")] * 60
    alerts, states, _ = _run_session(frames)
    assert alerts == [], f"起身应冻结计时，却弹了 {alerts}"
    assert states[-1] == "unknown"


def test_recovery_resets_silently():
    """低头 20s（已过触发）→ 正坐 20s：弹 1 次后恢复 NORMAL，不再弹。"""
    frames = [_frame("good")] * 10 + [_frame("drop")] * 20 + [_frame("good")] * 20
    alerts, states, _ = _run_session(frames)
    assert len(alerts) == 1
    assert states[-1] == "good"


if __name__ == "__main__":
    failures = 0
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError:
            failures += 1
            print(f"FAIL {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
