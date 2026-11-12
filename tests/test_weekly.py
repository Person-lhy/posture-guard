"""weekly.py 单测：逐日聚合、无数据占位、不良率/最长段计算、文本周报。

运行：python tests/test_weekly.py   （或 python -m pytest tests/）
出图层（render_chart）需 matplotlib，沙箱没有 → 自动跳过，不算失败。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import storage  # noqa: E402
import weekly  # noqa: E402


def _row(level, theta=0.0, drop=0.0):
    return {"theta_deg": theta, "head_drop": drop,
            "head_pitch_deg": 0.0, "shoulder_tilt": 0.0, "neck_ratio": 0.5}


def _tmp_env():
    d = tempfile.mkdtemp(prefix="pg_week_")
    old = storage.DATA_DIR
    storage.DATA_DIR = d
    return d, old


def _fill_day(day, levels, theta=28.0, drop=0.4):
    for i, lv in enumerate(levels):
        storage.append_row(lv, _row(lv, theta if lv != "good" else 5.0,
                                    drop if lv != "good" else 0.1),
                           timestamp=f"{day[:4]}-{day[4:6]}-{day[6:8]}T09:00:{i:02d}",
                           day=day)


def test_aggregate_empty_days_placeholder():
    d, old = _tmp_env()
    try:
        rows = weekly.aggregate(days=3, end="20260103")
        assert len(rows) == 3
        assert [r["day"] for r in rows] == ["20260101", "20260102", "20260103"]
        assert all(r["records"] == 0 and r["bad_pct"] == 0.0 for r in rows)
    finally:
        storage.DATA_DIR = old
        shutil.rmtree(d, ignore_errors=True)


def test_aggregate_bad_pct_and_longest():
    d, old = _tmp_env()
    try:
        _fill_day("20260101", ["good"] * 6 + ["bad"] * 4)      # 40% 不良
        _fill_day("20260102", ["good"] * 10)                    # 0% 不良
        rows = weekly.aggregate(days=2, end="20260102")
        assert rows[0]["bad_pct"] == 40.0
        assert rows[0]["longest_bad"] == 4                      # 连续 4 秒
        assert rows[0]["counts"]["bad"] == 4
        assert rows[1]["bad_pct"] == 0.0 and rows[1]["longest_bad"] == 0
    finally:
        storage.DATA_DIR = old
        shutil.rmtree(d, ignore_errors=True)


def test_aggregate_split_segments():
    """两段坏姿势中间隔好姿势 → 最长段取更长的那段。"""
    d, old = _tmp_env()
    try:
        _fill_day("20260101", ["bad"] * 3 + ["good"] * 5 + ["bad"] * 6)
        rows = weekly.aggregate(days=1, end="20260101")
        assert rows[0]["longest_bad"] == 6
    finally:
        storage.DATA_DIR = old
        shutil.rmtree(d, ignore_errors=True)


def test_build_text_no_data():
    d, old = _tmp_env()
    try:
        text, ok = weekly.build_text(days=3, end="20990103")
        assert not ok and "暂无数据" in text
    finally:
        storage.DATA_DIR = old
        shutil.rmtree(d, ignore_errors=True)


def test_build_text_content():
    d, old = _tmp_env()
    try:
        _fill_day("20260101", ["good"] * 6 + ["bad"] * 4, theta=30.0, drop=0.45)
        _fill_day("20260102", ["good"] * 10)
        text, ok = weekly.build_text(days=2, end="20260102")
        assert ok
        assert "01-01" in text and "01-02" in text
        assert "40.0%" in text
        assert "最差的一天" in text and "01-01" in text
        assert "本周没有" not in text          # 有 40% 的日子，不该出现"保持"
    finally:
        storage.DATA_DIR = old
        shutil.rmtree(d, ignore_errors=True)


def test_build_text_all_good_days():
    d, old = _tmp_env()
    try:
        _fill_day("20260101", ["good"] * 10)
        text, ok = weekly.build_text(days=1, end="20260101")
        assert ok and "本周没有不良率" in text
    finally:
        storage.DATA_DIR = old
        shutil.rmtree(d, ignore_errors=True)


def test_render_chart_skips_without_matplotlib():
    d, old = _tmp_env()
    try:
        _fill_day("20260101", ["good"] * 5 + ["bad"] * 5)
        out = weekly.render_chart(days=1, out_path=os.path.join(d, "w.png"), end="20260101")
        if out is None:
            print("      （matplotlib 不可用，已按设计跳过出图）")
        else:
            assert os.path.exists(out)
    finally:
        storage.DATA_DIR = old
        shutil.rmtree(d, ignore_errors=True)


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
