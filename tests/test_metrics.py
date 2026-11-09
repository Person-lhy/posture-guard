"""metrics.py 单测：合成关键点 → extract_metrics → 裸值断言。

运行：python -m pytest tests/ -q   （或 python tests/test_metrics.py）
不依赖 mediapipe —— landmark 用 duck typing 的假对象。
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import (  # noqa: E402
    LEFT_EAR,
    LEFT_SHOULDER,
    NOSE,
    RIGHT_EAR,
    RIGHT_SHOULDER,
    extract_metrics,
)


class FakeLM:
    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 1.0):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


def _lms(nose, ear_l, ear_r, sh_l, sh_r):
    """按 33 点约定摆放 5 个锚点，其余补零（不会被读到）。"""
    arr = [FakeLM(0.0, 0.0) for _ in range(33)]
    arr[NOSE] = nose
    arr[LEFT_EAR] = ear_l
    arr[RIGHT_EAR] = ear_r
    arr[LEFT_SHOULDER] = sh_l
    arr[RIGHT_SHOULDER] = sh_r
    return arr


def _close(a, b, tol=1e-6):
    assert a is not None and math.isclose(a, b, abs_tol=tol), f"{a} != {b}"


def test_upright_pose():
    # 640x480，肩宽 200px，直立：耳(320,200) 肩(320,300) 鼻(320,228) → d=28/200=0.14
    # z：鼻 z=50、耳 z=100（鼻更靠近相机）→ φ=atan2(0, 50)=0
    lms = _lms(FakeLM(0.5, 0.475, 50.0), FakeLM(0.4375, 5 / 12, 100.0),
               FakeLM(0.5625, 5 / 12, 100.0),
               FakeLM(0.34375, 0.625, 100.0), FakeLM(0.65625, 0.625, 100.0))
    m = extract_metrics(lms, 640, 480)
    _close(m["scale"], 200.0)
    _close(m["theta_deg"], 0.0)
    _close(m["head_drop"], 0.14)           # (228-200)/200
    _close(m["neck_ratio"], 0.5)           # 100/200
    _close(m["shoulder_tilt"], 0.0)
    _close(m["head_pitch_deg"], 0.0, 0.1)  # 鼻与耳同高，只在 z 上偏移；y 归一化有浮点残差，容差 0.1°
    _close(m["confidence"], 1.0)


def test_lean_and_drop():
    # 耳偏右 20px、比肩高 110px → θ=atan2(20,110)
    # 鼻 (340,280) 低于耳 (340,210) 70px → d=70/200=0.35
    lms = _lms(FakeLM(340 / 640, 280 / 480), FakeLM(340 / 640, 210 / 480),
               FakeLM(340 / 640, 210 / 480), FakeLM(220 / 640, 320 / 480),
               FakeLM(420 / 640, 320 / 480))
    m = extract_metrics(lms, 640, 480)
    _close(m["theta_deg"], math.degrees(math.atan2(20.0, 110.0)), 1e-6)
    _close(m["head_drop"], 70.0 / 200.0)
    _close(m["neck_ratio"], math.hypot(20.0, 110.0) / 200.0, 1e-6)


def test_shoulder_tilt_slope():
    # 左肩高 15px、肩距 200px → s=0.075
    lms = _lms(FakeLM(0.5, 0.4), FakeLM(0.4375, 5 / 12), FakeLM(0.5625, 5 / 12),
               FakeLM(0.34375, 292.5 / 480), FakeLM(0.65625, 307.5 / 480))
    m = extract_metrics(lms, 640, 480)
    _close(m["shoulder_tilt"], 0.075, 1e-6)


def test_low_visibility_returns_none():
    # 鼻不可见 → 所有指标 None，但 confidence 如实报告
    lms = _lms(FakeLM(0.5, 0.4, visibility=0.3), FakeLM(0.4375, 5 / 12),
               FakeLM(0.5625, 5 / 12), FakeLM(0.34375, 0.625),
               FakeLM(0.65625, 0.625))
    m = extract_metrics(lms, 640, 480)
    assert m["theta_deg"] is None
    assert m["head_drop"] is None
    assert m["head_pitch_deg"] is None
    assert m["neck_ratio"] is None
    assert m["shoulder_tilt"] is None
    assert m["confidence"] == 0.3


def test_degenerate_scale_returns_none_for_ratios():
    # 双肩重合 → scale=0，无量纲指标 None；θ 仍可算（颈线非零）
    lms = _lms(FakeLM(0.5, 0.4), FakeLM(0.4375, 5 / 12), FakeLM(0.5625, 5 / 12),
               FakeLM(0.5, 0.625), FakeLM(0.5, 0.625))
    m = extract_metrics(lms, 640, 480)
    assert m["head_drop"] is None
    assert m["neck_ratio"] is None
    assert m["shoulder_tilt"] is None
    assert m["scale"] is None
    assert m["theta_deg"] == 0.0


def test_anchor_indices_match_mediapipe_spec():
    assert NOSE == 0
    assert (LEFT_EAR, RIGHT_EAR) == (7, 8)
    assert (LEFT_SHOULDER, RIGHT_SHOULDER) == (11, 12)


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
