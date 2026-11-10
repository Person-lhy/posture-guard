"""Day 7：合成数据生成器 —— 无摄像头也能验证全链路。

生成"正坐 + 某种坏姿势"的模拟会话 CSV，配合校准报告验证阈值建议逻辑。
**每种坏姿势单独出一份**：混合多种坏姿势会稀释单指标区分度
（比如纯低头的 theta 和正常重叠，混进池子会把 theta 建议阈值拉低 → 误报）。

用法：
    python make_demo_data.py              # 全部类型，逐个出 CSV + 报告
    python make_demo_data.py slouch       # 只生成"含胸前倾"类型
    python make_demo_data.py drop side    # 两种

类型：slouch（含胸前倾，theta 大）/ drop（纯低头，drop 大）
      side（侧歪，shldr 大）/ mixed（混合，演示区分度被稀释的坑）
"""

from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Optional

import storage

# 各坏姿势的裸值模板（与 metrics.extract_metrics 输出字段一致）
GOOD = {"theta_deg": 5.0, "head_drop": 0.10, "head_pitch_deg": 3.0,
        "shoulder_tilt": 0.03, "neck_ratio": 0.50}
BAD_TYPES = {
    "slouch": {"theta_deg": 30.0, "head_drop": 0.12, "head_pitch_deg": 6.0,
               "shoulder_tilt": 0.05, "neck_ratio": 0.42},   # theta 大，其余正常
    "drop":   {"theta_deg": 6.0, "head_drop": 0.45, "head_pitch_deg": 38.0,
               "shoulder_tilt": 0.03, "neck_ratio": 0.50},   # 低头指标大
    "side":   {"theta_deg": 32.0, "head_drop": 0.11, "head_pitch_deg": 4.0,
               "shoulder_tilt": 0.18, "neck_ratio": 0.48},   # 肩歪
}
NOISE = {"theta_deg": 1.5, "head_drop": 0.02, "head_pitch_deg": 1.5,
         "shoulder_tilt": 0.01, "neck_ratio": 0.02}

GOOD_SECONDS = 120
BAD_SECONDS = 90


def _jitter(base: Optional[float], key: str, rng: random.Random) -> Optional[float]:
    if base is None:
        return None
    return round(base + rng.gauss(0.0, NOISE[key]), 4)


def _emit(rows, level, tmpl, rng, start_sec, dur):
    sec = start_sec
    for _ in range(dur):
        values: Dict[str, Optional[float]] = {k: _jitter(v, k, rng) for k, v in tmpl.items()}
        hh, mm, ss = 9 + sec // 3600, (sec // 60) % 60, sec % 60
        ts = f"2026-01-01T{hh:02d}:{mm:02d}:{ss:02d}"
        conf = round(rng.uniform(0.85, 1.0), 2)
        rows.append((level, values, conf, ts))
        sec += 1
    return sec


def generate(bad_type: str, seed: int = 7) -> str:
    """生成指定坏姿势类型的会话 CSV（正坐 GOOD_SECONDS + 坏 BAD_SECONDS），返回路径。

    写入 storage.DATA_DIR（由调用方决定，main() 统一指向 data/demo/）。
    """
    rng = random.Random(seed)
    bad = BAD_TYPES.get(bad_type)
    day = f"20260101_{bad_type}"
    rows: List[tuple] = []
    sec = _emit(rows, "good", GOOD, rng, 0, GOOD_SECONDS)
    if bad_type == "mixed":
        for i, (name, tmpl) in enumerate(BAD_TYPES.items()):
            sec = _emit(rows, "bad", tmpl, rng, sec, 30)
    elif bad is not None:
        sec = _emit(rows, "bad", bad, rng, sec, BAD_SECONDS)
    else:
        raise ValueError(f"未知类型 {bad_type}，可选：{list(BAD_TYPES)} / mixed")

    os.makedirs(storage.DATA_DIR, exist_ok=True)
    for level, values, conf, ts in rows:
        storage.append_row(level, values, confidence=conf, timestamp=ts, day=day)
    return storage._csv_path(day)


def main() -> int:
    args = sys.argv[1:] or ["slouch", "drop", "side", "mixed"]
    demo_dir = os.path.join(storage.DATA_DIR, "demo")
    old = storage.DATA_DIR
    storage.DATA_DIR = demo_dir          # 读写同一目录，summary 才能找到 CSV
    try:
        for bad_type in args:
            path = generate(bad_type)
            day = os.path.splitext(os.path.basename(path))[0]
            rep = storage.summary(day=day,
                                  out_path=os.path.join(demo_dir, f"summary_{day}.txt"))
            print(f"[{bad_type}] {os.path.relpath(path, demo_dir)} → {os.path.relpath(rep, demo_dir)}")
    finally:
        storage.DATA_DIR = old
    return 0


if __name__ == "__main__":
    sys.exit(main())
