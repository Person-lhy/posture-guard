"""posture_rules.py + reminder_engine.py 单测。

运行：python -m pytest tests/ -q   （或 python tests/test_rules.py）
不依赖 mediapipe/cv2 —— 假时钟 + 合成裸值。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import THRESHOLDS  # noqa: E402
from posture_rules import Grade, MedianFilter, analyze, default_grades  # noqa: E402
from reminder_engine import (  # noqa: E402
    ALERTED,
    COUNTDOWN,
    NORMAL,
    SNOOZED,
    ReminderEngine,
)


def _state(level, reasons=(), text=""):
    class P:
        pass
    p = P()
    p.level, p.reasons, p.text = level, list(reasons), text
    return p


# ---------- Grade：迟滞 ----------

def test_grade_hysteresis_no_flapping():
    # warn=25, recover=20, severe=40
    g = Grade(25.0, 20.0, 40.0)
    assert g.update(10.0) == "good"
    assert g.update(26.0) == "warn"     # 越进线
    assert g.update(22.0) == "warn"     # 未回出线 → 维持 warn（不横跳）
    assert g.update(24.0) == "warn"
    assert g.update(19.0) == "good"     # 回到出线内
    assert g.update(22.0) == "good"     # good 侧中带不升级


def test_grade_severe_and_downgrade():
    g = Grade(25.0, 20.0, 40.0)
    g.update(45.0)
    assert g.level == "bad"
    assert g.update(30.0) == "warn"     # bad→warn 直降
    assert g.update(45.0) == "bad"
    assert g.update(15.0) == "good"     # bad→good 直升


def test_grade_none_freezes():
    g = Grade(25.0, 20.0, 40.0)
    g.update(26.0)
    assert g.update(None) == "warn"     # 未知不清零、不升级


def test_grade_no_severe_stays_warn():
    g = Grade(0.35, 0.28)               # 无 severe 档
    assert g.update(0.5) == "warn"
    assert g.update(0.2) == "good"


# ---------- MedianFilter ----------

def test_median_filter_rejects_spike():
    mf = MedianFilter(5)
    out = [mf.push("k", v) for v in (10.0, 10.0, 999.0, 10.0, 11.0)]
    assert out[0] is None and out[1] is None   # 样本不足
    assert out[2] == 10.0                      # 野值 999 被中值吃掉
    assert out[4] == 10.0


def test_median_filter_none_and_reset():
    mf = MedianFilter(5)
    assert mf.push("k", None) is None
    mf.push("k", 1.0); mf.push("k", 2.0); mf.push("k", 3.0)
    mf.reset()
    assert mf.push("k", 4.0) is None           # reset 后重新计样本


# ---------- analyze：合成 ----------

def _raw(**kw):
    base = {"theta_deg": 0.0, "head_drop": 0.1, "head_pitch_deg": 0.0,
            "neck_ratio": 0.5, "shoulder_tilt": 0.0, "confidence": 1.0}
    base.update(kw)
    return base


def test_analyze_good_posture():
    mf, gr = MedianFilter(5), default_grades()
    for _ in range(6):
        ps = analyze(_raw(), mf, gr)
    assert ps.level == "good"
    assert ps.reasons == []


def test_analyze_flags_theta():
    mf, gr = MedianFilter(5), default_grades()
    for _ in range(6):
        ps = analyze(_raw(theta_deg=28.0), mf, gr)
    assert ps.level == "warn"
    assert any("theta" in r for r in ps.reasons)
    assert ps.text.startswith("WARN")


def test_analyze_severe_theta():
    mf, gr = MedianFilter(5), default_grades()
    for _ in range(6):
        ps = analyze(_raw(theta_deg=45.0), mf, gr)
    assert ps.level == "bad"
    assert ps.text.startswith("BAD")


def test_analyze_holds_between_thresholds():
    mf, gr = MedianFilter(5), default_grades()
    for _ in range(6):
        analyze(_raw(theta_deg=28.0), mf, gr)      # 进 warn（中值=28）
    ps = analyze(_raw(theta_deg=22.0), mf, gr)      # 22 在 25/20 之间
    assert ps.level == "warn"                       # 迟滞：不横跳
    for _ in range(6):
        ps = analyze(_raw(theta_deg=15.0), mf, gr)  # 中值压到 15，跌破出线
    assert ps.level == "good"                       # 恢复需要连续好帧（滤波+迟滞）


def test_analyze_none_metrics_all_unknown():
    mf, gr = MedianFilter(5), default_grades()
    raw = _raw(theta_deg=None, head_drop=None, head_pitch_deg=None,
               neck_ratio=None, shoulder_tilt=None)
    for _ in range(6):
        ps = analyze(raw, mf, gr)
    assert ps.level == "unknown"


# ---------- ReminderEngine：假时钟 ----------

class FakeClock:
    def __init__(self):
        self.t = 0.0
    def __call__(self):
        return self.t
    def tick(self, dt):
        self.t += dt


def _engine(**kw):
    c = FakeClock()
    e = ReminderEngine(clock=c, **kw)
    return c, e


def test_engine_normal_to_alert():
    c, e = _engine(trigger_seconds=15.0)
    ps = _state("good")
    assert e.update(ps) is None and e.state == NORMAL
    e.update(_state("warn", ["theta +28.0"], ""))
    assert e.state == COUNTDOWN
    c.tick(10.0)
    assert e.update(_state("warn", ["theta +28.0"], "")) is None
    c.tick(6.0)                                        # 累计 16s ≥ 15s
    act = e.update(_state("warn", ["theta +28.0"], ""))
    assert act and act["type"] == "alert" and act["level"] == "warn"
    assert e.state == ALERTED


def test_engine_recovers_after_good_hold():
    c, e = _engine(trigger_seconds=15.0, recover_seconds=5.0)
    e.update(_state("bad", ["theta +45.0"], ""))
    c.tick(20.0)
    e.update(_state("bad", ["theta +45.0"], ""))       # 触发 alert
    assert e.state == ALERTED
    e.update(_state("good", [], ""))                   # 回好，但未满 5s
    c.tick(3.0)
    e.update(_state("good", [], ""))
    assert e.state == ALERTED                          # 迟滞：还没算恢复
    c.tick(3.0)
    e.update(_state("good", [], ""))
    assert e.state == NORMAL


def test_engine_cooldown_realert():
    c, e = _engine(trigger_seconds=5.0, cooldown_seconds=300.0)
    e.update(_state("warn", ["x"], ""))
    c.tick(6.0)
    act1 = e.update(_state("warn", ["x"], ""))
    assert act1 and act1["type"] == "alert"
    c.tick(100.0)
    assert e.update(_state("warn", ["x"], "")) is None  # 冷却中
    c.tick(201.0)
    act2 = e.update(_state("warn", ["x"], ""))
    assert act2 and act2["type"] == "alert"


def test_engine_escalates_to_severe():
    c, e = _engine(trigger_seconds=5.0, escalate_seconds=600.0)
    e.update(_state("warn", ["x"], ""))
    c.tick(6.0)
    act = e.update(_state("warn", ["x"], ""))
    assert act["level"] == "warn"
    c.tick(600.0)                                      # 越过冷却，累计 > 600s
    act = e.update(_state("warn", ["x"], ""))
    assert act["level"] == "severe"
    assert "停下休息" in act["text"]


def test_engine_unknown_freezes():
    c, e = _engine(trigger_seconds=15.0)
    e.update(_state("warn", ["x"], ""))
    c.tick(10.0)
    act = e.update(_state("unknown", [], ""))          # 人离开
    assert act == {"type": "silence"}
    assert e.state == NORMAL and e.bad_since is None   # 计时清零，不把离开工位算进坏姿势
    c.tick(1000.0)
    act = e.update(_state("warn", ["x"], ""))          # 回来后重新计时
    assert act is None and e.state == COUNTDOWN
    c.tick(14.0)
    assert e.update(_state("warn", ["x"], "")) is None # 新计时不满 15s
    c.tick(2.0)
    assert e.update(_state("warn", ["x"], ""))["type"] == "alert"


def test_engine_snooze():
    c, e = _engine(trigger_seconds=5.0, snooze_seconds=1800.0)
    e.update(_state("warn", ["x"], ""))
    c.tick(6.0)
    assert e.update(_state("warn", ["x"], ""))["type"] == "alert"
    e.snooze()
    assert e.state == SNOOZED
    c.tick(100.0)
    assert e.update(_state("warn", ["x"], "")) is None  # 静默
    c.tick(1800.0)
    e.update(_state("good", [], ""))                    # 期满 → NORMAL
    assert e.state == NORMAL


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
