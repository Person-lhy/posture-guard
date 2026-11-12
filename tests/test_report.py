"""report.py 单测：占比统计、时间线分桶、坏姿势 TOP 时段排序与主因。

运行：python tests/test_report.py   （或 python -m pytest tests/）
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import report  # noqa: E402
import storage  # noqa: E402


def _tmpdir():
    return tempfile.mkdtemp(prefix="pg_rep_")


def _row(level, ts, theta=0.0, drop=0.0):
    return {"theta_deg": theta, "head_drop": drop,
            "head_pitch_deg": 0.0, "shoulder_tilt": 0.0, "neck_ratio": 0.5}, level, ts


def _write(rows, day):
    for values, level, ts in rows:
        storage.append_row(level, values, timestamp=ts, day=day)


def test_empty_report():
    d = _tmpdir()
    old = storage.DATA_DIR
    storage.DATA_DIR = d
    try:
        text, ok = report.build_report("20990101")
        assert not ok and "暂无数据" in text
    finally:
        storage.DATA_DIR = old
        shutil.rmtree(d, ignore_errors=True)


def test_counts_and_percentages():
    d = _tmpdir()
    old = storage.DATA_DIR
    storage.DATA_DIR = d
    try:
        rows = []
        for i in range(60):
            rows.append(_row("good", f"2026-11-12T09:00:{i:02d}"))
        for i in range(30):
            rows.append(_row("warn", f"2026-11-12T09:01:{i:02d}", theta=28.0))
        for i in range(10):
            rows.append(_row("bad", f"2026-11-12T09:02:{i:02d}", theta=45.0))
        _write(rows, "20261112")
        text, ok = report.build_report("20261112")
        assert ok
        assert "记录 100 秒" in text
        assert "60.0%" in text          # good 占比
        assert "30.0%" in text          # warn 占比
        assert "10.0%" in text          # bad 占比
    finally:
        storage.DATA_DIR = old
        shutil.rmtree(d, ignore_errors=True)


def test_timeline_has_hours():
    d = _tmpdir()
    old = storage.DATA_DIR
    storage.DATA_DIR = d
    try:
        rows = []
        for i in range(20):
            rows.append(_row("good", f"2026-11-12T09:00:{i:02d}"))
        for i in range(20):
            rows.append(_row("bad", f"2026-11-12T14:00:{i:02d}", drop=0.5))
        _write(rows, "20261112")
        text, _ = report.build_report("20261112")
        assert "09:00 |" in text
        assert "14:00 |" in text
    finally:
        storage.DATA_DIR = old
        shutil.rmtree(d, ignore_errors=True)


def test_bad_segments_ranked_by_duration():
    d = _tmpdir()
    old = storage.DATA_DIR
    storage.DATA_DIR = d
    try:
        rows = []
        # 短段 4s（warn）→ 排第 2；长段 10s（bad）→ 排第 1
        for i in range(4):
            rows.append(_row("warn", f"2026-11-12T09:00:{i:02d}", theta=28.0))
        for i in range(10):
            rows.append(_row("bad", f"2026-11-12T10:00:{i:02d}", drop=0.5))
        # 好姿势断段
        rows.append(_row("good", "2026-11-12T10:00:59"))
        _write(rows, "20261112")
        text, _ = report.build_report("20261112")
        assert "共 10 秒" in text
        assert "共 4 秒" in text
        assert text.index("共 10 秒") < text.index("共 4 秒")   # 长段排前面
    finally:
        storage.DATA_DIR = old
        shutil.rmtree(d, ignore_errors=True)


def test_main_reason_labels():
    d = _tmpdir()
    old = storage.DATA_DIR
    storage.DATA_DIR = d
    try:
        rows = []
        for i in range(6):
            rows.append(_row("bad", f"2026-11-12T09:00:{i:02d}", drop=0.5))
        _write(rows, "20261112")
        text, _ = report.build_report("20261112")
        assert "主因：低头" in text
    finally:
        storage.DATA_DIR = old
        shutil.rmtree(d, ignore_errors=True)


def test_min_length_filter():
    """单秒毛刺（<3s）不该进 TOP 段。"""
    d = _tmpdir()
    old = storage.DATA_DIR
    storage.DATA_DIR = d
    try:
        rows = [_row("bad", "2026-11-12T09:00:00", drop=0.5),
                _row("good", "2026-11-12T09:00:01"),
                _row("good", "2026-11-12T09:00:02")]
        _write(rows, "20261112")
        text, _ = report.build_report("20261112")
        assert "没有持续的坏姿势" in text
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
