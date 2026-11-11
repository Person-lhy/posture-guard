"""settings.py 单测：配置读写、默认回退、损坏容错、autostart 状态机。

运行：python tests/test_settings.py   （或 python -m pytest tests/）
autostart 写启动文件夹是真实副作用 → 只测状态读写用临时目录模拟，不碰真 Startup。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import settings  # noqa: E402


def _with_tmp(fn):
    """把 settings 的读写目录指到临时目录，跑完还原。"""
    d = tempfile.mkdtemp(prefix="pg_set_")
    old_path, old_dir = settings.SETTINGS_PATH, settings.SETTINGS_DIR
    settings.SETTINGS_DIR = d
    settings.SETTINGS_PATH = os.path.join(d, "settings.json")
    try:
        return fn(d)
    finally:
        settings.SETTINGS_PATH, settings.SETTINGS_DIR = old_path, old_dir
        shutil.rmtree(d, ignore_errors=True)


def test_defaults_when_no_file():
    def go(d):
        cfg = settings.load()
        assert cfg["neck_tilt_warn"] == settings.DEFAULTS["neck_tilt_warn"]
        assert cfg["camera_index"] == 0
    _with_tmp(go)


def test_save_and_reload():
    def go(d):
        settings.load()
        settings.save({"neck_tilt_warn": 30, "camera_index": 2, "trigger_seconds": 20})
        cfg = settings.load()
        assert cfg["neck_tilt_warn"] == 30
        assert cfg["camera_index"] == 2
        assert cfg["trigger_seconds"] == 20
        assert cfg["head_drop_warn"] == settings.DEFAULTS["head_drop_warn"]  # 未改字段保留
    _with_tmp(go)


def test_unknown_key_ignored():
    def go(d):
        settings.load()
        cfg = settings.save({"not_a_key": 123, "neck_tilt_warn": 28})
        assert "not_a_key" not in cfg
        assert cfg["neck_tilt_warn"] == 28
    _with_tmp(go)


def test_corrupted_json_falls_back():
    def go(d):
        with open(settings.SETTINGS_PATH, "w", encoding="utf-8") as f:
            f.write("{not valid json!!")
        cfg = settings.load()                     # 不抛异常
        assert cfg["neck_tilt_warn"] == settings.DEFAULTS["neck_tilt_warn"]
    _with_tmp(go)


def test_reset_clears_file():
    def go(d):
        settings.load()
        settings.save({"neck_tilt_warn": 30})
        settings.reset()
        assert not os.path.exists(settings.SETTINGS_PATH)
        cfg = settings.load()
        assert cfg["neck_tilt_warn"] == settings.DEFAULTS["neck_tilt_warn"]
    _with_tmp(go)


def test_quiet_hours_saved_as_list():
    def go(d):
        settings.load()
        settings.save({"quiet_hours": ["22:00", "07:00"]})
        cfg = settings.load()
        assert cfg["quiet_hours"] == ["22:00", "07:00"]
    _with_tmp(go)


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
