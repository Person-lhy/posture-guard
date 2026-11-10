"""storage.py 单测：CSV 落盘 + 阈值校准报告的分位数计算。

运行：python tests/test_storage.py   （或 python -m pytest tests/）
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import storage  # noqa: E402
from config import THRESHOLDS  # noqa: E402


def _tmpdir():
    d = tempfile.mkdtemp(prefix="pg_test_")
    return d


def _row(level, theta, drop, **kw):
    return {"theta_deg": theta, "head_drop": drop,
            "head_pitch_deg": kw.get("pitch", 0.0),
            "shoulder_tilt": kw.get("shldr", 0.0),
            "neck_ratio": kw.get("neck", 0.5)}


def test_append_row_creates_csv_with_header():
    d = _tmpdir()
    old = storage.DATA_DIR
    storage.DATA_DIR = d
    try:
        path = storage.append_row("good", _row("good", 5.0, 0.1), confidence=0.9)
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["level"] == "good"
        assert float(rows[0]["theta_deg"]) == 5.0
        assert float(rows[0]["confidence"]) == 0.9
        storage.append_row("warn", _row("warn", 28.0, 0.4), confidence=0.8)
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2                       # 追加而非覆盖
    finally:
        storage.DATA_DIR = old
        shutil.rmtree(d, ignore_errors=True)


def test_summary_percentiles_and_recommendations():
    d = _tmpdir()
    old = storage.DATA_DIR
    storage.DATA_DIR = d
    try:
        # 好姿势：theta 3~7（P85≈6.4），坏姿势：theta 28~32（P15≈28.4）
        for i in range(40):
            storage.append_row("good", _row("good", 3.0 + i * 0.1, 0.05 + i * 0.002),
                               timestamp=f"2026-11-10T09:00:{i:02d}", day="20261110")
        for i in range(40):
            storage.append_row("bad", _row("bad", 28.0 + i * 0.1, 0.40 + i * 0.003),
                               timestamp=f"2026-11-10T10:00:{i:02d}", day="20261110")
        path = storage.summary(day="20261110")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        assert "样本数 80" in text
        assert "好 40" in text and "不良 40" in text
        assert "theta" in text and "drop" in text
        # 建议阈值行必须出现（样本≥20），且锚定分布边界
        assert "warn     →" in text
        assert "recover  →" in text
        # theta：好 3~7（P85≈6.4）vs 坏 28~32（P15≈28.4），有区分度
        # warn ≈ 28.4（不良P15），recover ≈ 6.4（正常P85）
        assert "28." in text and "6." in text
    finally:
        storage.DATA_DIR = old
        shutil.rmtree(d, ignore_errors=True)


def test_summary_without_data():
    d = _tmpdir()
    old = storage.DATA_DIR
    storage.DATA_DIR = d
    try:
        path = storage.summary(day="20990101")      # 不存在的日期
        with open(path, encoding="utf-8") as f:
            text = f.read()
        assert "无数据文件" in text
    finally:
        storage.DATA_DIR = old
        shutil.rmtree(d, ignore_errors=True)


def test_summary_signed_theta_uses_abs():
    d = _tmpdir()
    old = storage.DATA_DIR
    storage.DATA_DIR = d
    try:
        # θ 负值（头偏左）也应计入 |θ| 分布，不能把 P50 拉成负数
        for i in range(30):
            storage.append_row("good", _row("good", -5.0 - i * 0.1, 0.1),
                               timestamp=f"2026-11-10T09:00:{i:02d}", day="20261110")
        for i in range(30):
            storage.append_row("bad", _row("bad", -30.0 - i * 0.1, 0.4),
                               timestamp=f"2026-11-10T10:00:{i:02d}", day="20261110")
        # 直接断言分位数逻辑：|θ| 的正常 P50 应为正数 ≈6.45，不是 -6.45
        import csv as _csv
        with open(storage._csv_path("20261110"), encoding="utf-8") as f:
            vals = [abs(float(r["theta_deg"])) for r in _csv.DictReader(f)
                    if r["level"] == "good"]
        assert storage._quantile(vals, 0.5) > 5.0, "负 θ 未取绝对值"
        # 报告层同样验证：正常 P50 列应为正
        path = storage.summary(day="20261110")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        theta_line = [ln for ln in text.splitlines() if ln.startswith("theta")]
        assert theta_line, text
        p50 = float(theta_line[0].split()[1])
        assert p50 > 0, f"θ 正常 P50 应为正，得到 {p50}"
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
