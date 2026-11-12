"""autotune.py 单测：建议阈值计算、无区分度跳过、单侧建议、落盘夹紧。

运行：python tests/test_autotune.py   （或 python -m pytest tests/）
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import autotune  # noqa: E402
import settings  # noqa: E402
import storage  # noqa: E402


def _row(level, theta, drop, shldr=0.03):
    """一行 CSV 记录（含 level 字段，与 storage.append_row 落盘结构一致）。"""
    return {"level": level,
            "theta_deg": theta, "head_drop": drop,
            "head_pitch_deg": 0.0, "shoulder_tilt": shldr, "neck_ratio": 0.5}


def _tmp_env():
    """临时 data 目录 + 临时 settings，跑完还原。"""
    d = tempfile.mkdtemp(prefix="pg_tune_")
    old_data, old_set_dir, old_set_path = storage.DATA_DIR, settings.SETTINGS_DIR, settings.SETTINGS_PATH
    storage.DATA_DIR = d
    settings.SETTINGS_DIR = d
    settings.SETTINGS_PATH = os.path.join(d, "settings.json")
    settings.load()
    return d, old_data, old_set_dir, old_set_path


def _restore(d, old_data, old_set_dir, old_set_path):
    storage.DATA_DIR = old_data
    settings.SETTINGS_DIR, settings.SETTINGS_PATH = old_set_dir, old_set_path
    shutil.rmtree(d, ignore_errors=True)


def test_separated_gives_both_sides():
    d, *old = _tmp_env()
    try:
        rows = ([_row("good", 3.0 + i * 0.1, 0.05 + i * 0.002) for i in range(40)]
                + [_row("bad", 28.0 + i * 0.1, 0.40 + i * 0.003) for i in range(40)])
        s = autotune.compute_suggestions(rows)
        assert "neck_tilt_warn" in s and "neck_tilt_recover" in s
        assert s["neck_tilt_warn"] > s["neck_tilt_recover"]      # warn 必须在 recover 之上
        assert 28.0 <= s["neck_tilt_warn"] <= 29.0              # ≈ bad P15
        assert 6.0 <= s["neck_tilt_recover"] <= 7.0             # ≈ good P85
        assert s["head_drop_warn"] > s["head_drop_recover"]
    finally:
        _restore(d, *old)


def test_overlapping_gives_nothing():
    """theta 正常/不良重叠 → 不给 theta 建议（给就是误报源）。"""
    d, *old = _tmp_env()
    try:
        rows = ([_row("good", 5.0 + i * 0.05, 0.05) for i in range(40)]
                + [_row("bad", 5.5 + i * 0.05, 0.40) for i in range(40)])   # theta 重叠，drop 分离
        s = autotune.compute_suggestions(rows)
        assert "neck_tilt_warn" not in s
        assert "head_drop_warn" in s                                # drop 有区分度，照给
    finally:
        _restore(d, *old)


def test_only_good_gives_recover_only():
    d, *old = _tmp_env()
    try:
        rows = [_row("good", 5.0 + i * 0.1, 0.05 + i * 0.002) for i in range(40)]
        s = autotune.compute_suggestions(rows)
        assert "neck_tilt_recover" in s
        assert "neck_tilt_warn" not in s
    finally:
        _restore(d, *old)


def test_insufficient_samples_gives_nothing():
    d, *old = _tmp_env()
    try:
        rows = [_row("good", 5.0, 0.05)] * 5 + [_row("bad", 30.0, 0.5)] * 5
        s = autotune.compute_suggestions(rows)
        assert s == {}
    finally:
        _restore(d, *old)


def test_apply_writes_and_clamps():
    """suggest 出 warn < 现存 recover 时，落盘必须夹紧，不能让 Grade 断言崩。"""
    d, *old = _tmp_env()
    try:
        settings.save({"neck_tilt_recover": 30.0})        # 人为设一个过高的 recover
        cfg = autotune.apply({"neck_tilt_warn": 10.0})    # warn 10 < recover 30 → 应夹紧
        assert cfg["neck_tilt_recover"] <= cfg["neck_tilt_warn"] == 10.0
    finally:
        _restore(d, *old)


def test_abs_theta_for_signed_values():
    """负 θ（头偏左）同样按绝对值参与分位数，建议值应为正。"""
    d, *old = _tmp_env()
    try:
        rows = ([_row("good", -5.0 - i * 0.1, 0.05) for i in range(40)]
                + [_row("bad", -30.0 - i * 0.1, 0.40) for i in range(40)])
        s = autotune.compute_suggestions(rows)
        assert s["neck_tilt_warn"] > 0 and s["neck_tilt_recover"] > 0
    finally:
        _restore(d, *old)


def test_suggest_from_day_reads_csv():
    d, *old = _tmp_env()
    try:
        for i in range(40):
            storage.append_row("good", _row("good", 3.0 + i * 0.1, 0.05),
                               timestamp=f"2026-11-12T09:00:{i:02d}", day="20261112")
        for i in range(40):
            storage.append_row("bad", _row("bad", 28.0 + i * 0.1, 0.40),
                               timestamp=f"2026-11-12T10:00:{i:02d}", day="20261112")
        s = autotune.suggest_from_day("20261112")
        assert "neck_tilt_warn" in s
    finally:
        _restore(d, *old)


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
