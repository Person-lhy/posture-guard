"""sedentary.py 单测：坐满提醒、离座清零、冷却防轰炸。

运行：python tests/test_sedentary.py   （或 python -m pytest tests/）
纯逻辑 + 假时钟，不依赖 Qt/cv2。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sedentary import SedentaryTracker  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 0.0
    def __call__(self):
        return self.t
    def tick(self, dt):
        self.t += dt


def _tracker(threshold=45 * 60.0, cooldown=15 * 60.0):
    c = FakeClock()
    return c, SedentaryTracker(threshold_seconds=threshold, cooldown_seconds=cooldown, clock=c)


def test_no_alert_before_threshold():
    c, t = _tracker(threshold=60.0)
    assert t.update("good") is None          # 刚坐下
    c.tick(59.0)
    assert t.update("good") is None          # 未满 60s
    c.tick(1.0)                              # 满 60s
    act = t.update("good")
    assert act and act["type"] == "sedentary"
    assert "1 分" in act["text"] or "60 秒" in act["text"]


def test_standing_resets_timer():
    c, t = _tracker(threshold=60.0)
    t.update("good")
    c.tick(50.0)
    t.update("good")
    assert t.update("unknown") is None       # 起身
    assert t.sitting_seconds() == 0.0
    t.update("good")                         # 重新坐下
    c.tick(50.0)
    assert t.update("good") is None          # 新计时不满 60s，不因上次残留误报
    c.tick(11.0)
    assert t.update("good")["type"] == "sedentary"


def test_cooldown_prevents_spam():
    c, t = _tracker(threshold=60.0, cooldown=120.0)
    t.update("good")
    c.tick(61.0)
    assert t.update("good")["type"] == "sedentary"   # 首次提醒
    c.tick(10.0)
    assert t.update("good") is None                  # 冷却中，不重复
    c.tick(111.0)                                    # 冷却结束
    assert t.update("good")["type"] == "sedentary"   # 再提醒


def test_warn_counts_as_sitting():
    """warn/bad 都算'坐着'——姿势不好不等于离座。"""
    c, t = _tracker(threshold=60.0)
    t.update("warn")
    c.tick(61.0)
    assert t.update("bad")["type"] == "sedentary"


def test_unknown_does_not_accumulate():
    c, t = _tracker(threshold=60.0)
    t.update("good")
    c.tick(30.0)
    for _ in range(20):                              # 长时间不在
        c.tick(1.0)
        assert t.update("unknown") is None
    assert t.sitting_seconds() == 0.0
    t.update("good")
    c.tick(59.0)
    assert t.update("good") is None


def test_sitting_seconds_tracks_elapsed():
    c, t = _tracker(threshold=3600.0)
    t.update("good")
    c.tick(100.0)
    assert abs(t.sitting_seconds() - 100.0) < 1e-6


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
