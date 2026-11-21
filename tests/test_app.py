"""app.py 里可离线测的纯逻辑：静默时段判定。

运行：python tests/test_app.py    （或 python -m pytest tests/）
说明：app.py 顶部 import PySide6/cv2，缺依赖时本文件自动跳过（打印 SKIP，退出码 0）。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import time
    from app import Worker
except Exception:                                  # pragma: no cover
    Worker = None


def _require_worker() -> None:
    """缺 PySide6/cv2 时跳过：pytest 认 unittest.SkipTest 为 SKIPPED（不是失败）。"""
    if Worker is None:
        raise unittest.SkipTest("PySide6/cv2 不可用，app.py 单测跳过")


def _quiet_at(hhmm: str, quiet_hours=("23:00", "08:00")) -> bool:
    """把 time.strftime 钉在指定时刻，调 Worker._quiet_now(rem)。"""
    hh, mm = map(int, hhmm.split(":"))
    orig = time.strftime
    time.strftime = lambda fmt, _t=(hh, mm): f"{_t[0]:02d}:{_t[1]:02d}"
    try:
        return Worker._quiet_now({"quiet_hours": list(quiet_hours)})
    finally:
        time.strftime = orig


def test_quiet_hours_overnight_window():
    """23:00–08:00 跨夜窗口：边界内外各测几点。"""
    _require_worker()
    for hhmm, expect in [("22:59", False), ("23:00", True), ("23:30", True),
                         ("00:00", True), ("07:59", True), ("08:00", False),
                         ("12:00", False)]:
        got = _quiet_at(hhmm)
        assert got is expect, f"{hhmm}: want {expect}, got {got}"


def test_quiet_hours_daytime_window():
    """白天窗口（如 12:00–13:00）不跨夜的写法也应工作。"""
    _require_worker()
    assert _quiet_at("12:30", ("12:00", "13:00")) is True
    assert _quiet_at("13:30", ("12:00", "13:00")) is False


if __name__ == "__main__":
    if Worker is None:
        print("SKIP: PySide6/cv2 不可用，app.py 单测跳过")
        sys.exit(0)
    failures = 0
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except unittest.SkipTest:
            print(f"SKIP {name}")
        except AssertionError:
            failures += 1
            print(f"FAIL {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
