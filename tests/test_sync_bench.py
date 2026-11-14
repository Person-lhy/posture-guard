"""sync_bench.py 单测：表格渲染与 README 标记段替换。

运行：python tests/test_sync_bench.py   （或 python -m pytest tests/）
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sync_bench  # noqa: E402


def _data():
    return {
        "generated": "2026-11-15T10:00:00",
        "results": [
            {"model": "MediaPipe Pose", "frames": 100, "detected": 96,
             "detection_rate": 0.96, "mean_ms": 12.5, "p50_ms": 12.0,
             "p95_ms": 18.0, "fps": 80.0, "theta_std": 1.234, "drop_std": 0.012},
            {"model": "YOLO11n-Pose", "frames": 100, "detected": 90,
             "detection_rate": 0.9, "mean_ms": 25.0, "p50_ms": 24.0,
             "p95_ms": 33.0, "fps": 40.0, "theta_std": None, "drop_std": None},
        ],
        "skipped": ["某模型：依赖未安装"],
    }


def test_render_table():
    t = sync_bench.render_table(_data())
    assert "MediaPipe Pose" in t and "YOLO11n-Pose" in t
    assert "96.0%" in t and "90.0%" in t
    assert "1.234" in t and "| – |" in t                      # None → –
    assert "跳过：某模型：依赖未安装" in t
    assert "角度一致性 ≠ 精度" in t                            # 口径必须带上


def test_update_readme_replaces_marker_section():
    text = ("前文\n" + sync_bench.START + "\n旧内容\n" + sync_bench.END + "\n后文")
    out = sync_bench.update_readme(text, "新表格")
    assert "旧内容" not in out and "新表格" in out
    assert out.startswith("前文") and out.endswith("后文")      # 标记外不动


def test_update_readme_without_markers_unchanged():
    text = "没有标记段"
    assert sync_bench.update_readme(text, "新表格") == text


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
