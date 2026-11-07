"""geometry.py 纯函数单测。

运行：python -m pytest tests/ -q   （或 python tests/test_geometry.py 直接跑）
不依赖任何第三方库 —— geometry.py 本身只用标准库 math。
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry import (  # noqa: E402
    head_drop_ratio,
    head_pitch_deg,
    midpoint,
    neck_length_ratio,
    neck_tilt_deg,
    shoulder_tilt,
    to_pixels,
)


class FakeLM:
    """冒充 MediaPipe landmark（duck typing：只用 .x/.y/.z）。"""

    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x, self.y, self.z = x, y, z


# ---------- to_pixels ----------

def test_to_pixels_basic():
    px, py, pz = to_pixels(FakeLM(0.5, 0.5), 1280, 720)
    assert px == 640 and py == 360

def test_to_pixels_z_scales_by_width():
    # z 与 x 同尺度（MediaPipe 约定），乘 width 而非 height
    _, _, pz = to_pixels(FakeLM(0.0, 0.0, 0.25), 800, 600)
    assert pz == 200.0

# ---------- midpoint ----------

def test_midpoint():
    assert midpoint((0.0, 0.0), (10.0, 4.0)) == (5.0, 2.0)

# ---------- neck_tilt_deg ----------

def test_neck_tilt_vertical_is_zero():
    # 耳在肩正上方（图像系 y 向下：耳 y 更小）
    theta = neck_tilt_deg((100.0, 100.0), (100.0, 200.0))
    assert theta == 0.0

def test_neck_tilt_45_degrees_positive():
    # 耳在肩右上方，与竖直成 45°
    theta = neck_tilt_deg((200.0, 100.0), (100.0, 200.0))
    assert math.isclose(theta, 45.0, abs_tol=1e-9)

def test_neck_tilt_negative_left():
    # 耳在肩左上方 → 负角
    theta = neck_tilt_deg((100.0, 100.0), (200.0, 200.0))
    assert math.isclose(theta, -45.0, abs_tol=1e-9)

def test_neck_tilt_degenerate_returns_none():
    assert neck_tilt_deg((100.0, 100.0), (100.0, 100.0)) is None

# ---------- head_drop_ratio ----------

def test_head_drop_positive_when_nose_below_ears():
    # 鼻比耳低 30px，肩宽 100px → 0.3
    d = head_drop_ratio((100.0, 130.0), (100.0, 100.0), 100.0)
    assert math.isclose(d, 0.3, abs_tol=1e-9)

def test_head_drop_negative_when_nose_above():
    d = head_drop_ratio((100.0, 90.0), (100.0, 100.0), 100.0)
    assert math.isclose(d, -0.1, abs_tol=1e-9)

def test_head_drop_zero_scale_returns_none():
    assert head_drop_ratio((0.0, 10.0), (0.0, 0.0), 0.0) is None

# ---------- head_pitch_deg ----------

def test_head_pitch_level_is_zero():
    # 鼻与耳同高、只在 z 上偏移 → 俯仰角 0
    phi = head_pitch_deg((0.0, 100.0, -20.0), (0.0, 100.0, 0.0))
    assert math.isclose(phi, 0.0, abs_tol=1e-9)

def test_head_pitch_down_is_positive():
    # 鼻相对耳下移 10、前移 10 → 45°
    phi = head_pitch_deg((0.0, 110.0, -10.0), (0.0, 100.0, 0.0))
    assert math.isclose(phi, 45.0, abs_tol=1e-9)

def test_head_pitch_degenerate_returns_none():
    assert head_pitch_deg((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) is None

# ---------- neck_length_ratio ----------

def test_neck_length_ratio():
    # 颈线长 50，肩宽 100 → 0.5
    r = neck_length_ratio((100.0, 100.0), (100.0, 150.0), 100.0)
    assert math.isclose(r, 0.5, abs_tol=1e-9)

def test_neck_length_ratio_zero_scale_returns_none():
    assert neck_length_ratio((0.0, 0.0), (0.0, 10.0), 0.0) is None

# ---------- shoulder_tilt ----------

def test_shouldoulder_tilt_flat_shoulders():
    s = shoulder_tilt((50.0, 100.0), (150.0, 100.0))
    assert s == 0.0

def test_shoulder_tilt_slope():
    # 左肩比右肩高 15px，肩距 100px → 0.15
    s = shoulder_tilt((50.0, 92.5), (150.0, 107.5))
    assert math.isclose(s, 0.15, abs_tol=1e-9)

def test_shoulder_tilt_vertical_returns_none():
    # 双肩 x 相同（退化），无法定义水平差
    assert shoulder_tilt((100.0, 90.0), (100.0, 110.0)) is None


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
