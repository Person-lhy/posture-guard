"""Day 10（M2-3）：每日报告 —— 当天姿势时间线 + 坏姿势 TOP 时段。

从当天 CSV（storage 写的那份）读出逐秒判级，汇总成三块：
  1. 总览：好/警告/不良/未知 占比
  2. 24 小时时间线：每小时好坏分布（文本条形图，按实际有数据的时段）
  3. 坏姿势 TOP3 时段：连续不良/警告段按时长排名，附主因

纯 stdlib，输入 CSV 路径或 day 字符串，返回文本报告（可直接弹窗显示）。
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import storage

LEVELS = ("good", "warn", "bad", "unknown")
LEVEL_MARK = {"good": "·", "warn": "w", "bad": "X", "unknown": "?"}


def _load(day: Optional[str]) -> List[dict]:
    path = storage._csv_path(day)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _hour(ts: str) -> str:
    # ISO 时间戳 "2026-11-12T09:23:45" → "09"
    return ts[11:13] if len(ts) >= 13 else "??"


def _hour_ranges(rows: List[dict]) -> List[str]:
    """有数据的小时列表（升序去重）。"""
    return sorted({r["timestamp"][11:13] for r in rows if len(r["timestamp"]) >= 13})


def _ts_seconds(ts: str) -> Optional[float]:
    """ISO 时间戳 → 当日秒数（粗略，用于判断时间连续性）。"""
    try:
        return int(ts[11:13]) * 3600 + int(ts[14:16]) * 60 + int(ts[17:19])
    except (ValueError, IndexError):
        return None


def _bad_segments(rows: List[dict], min_len: int = 3) -> List[dict]:
    """连续 warn/bad 段（≥min_len 秒），按时长降序。

    段合并规则：level∈{warn,bad} 且与上一秒时间连续才并入；
    时间断裂（中间没记录/隔了很久）或遇到好姿势/unknown 都断段。
    """
    segs: List[dict] = []
    cur: Optional[dict] = None
    prev_sec: Optional[float] = None

    def close():
        nonlocal cur
        if cur is not None:
            segs.append(cur)
            cur = None

    for r in rows:
        sec = _ts_seconds(r["timestamp"])
        continuous = (sec is not None and prev_sec is not None
                      and 0 < sec - prev_sec <= 2.0)   # 1s 采样，容忍 2s 内的缝隙
        if r["level"] in ("warn", "bad"):
            if cur is None or not continuous:
                close()
                cur = {"start": r["timestamp"], "end": r["timestamp"],
                       "levels": Counter(), "reasons": Counter()}
            cur["end"] = r["timestamp"]
            cur["levels"][r["level"]] += 1
            for k in ("theta_deg", "head_drop", "head_pitch_deg", "shoulder_tilt"):
                try:
                    v = float(r.get(k) or 0)
                except ValueError:
                    continue
                if abs(v) > 0:
                    cur["reasons"][k] += 1
        else:
            close()
        if sec is not None:
            prev_sec = sec
    close()

    def dur(seg: dict) -> int:
        return sum(seg["levels"].values())

    out = [s for s in segs if dur(s) >= min_len]
    out.sort(key=dur, reverse=True)
    return out


def _main_reason(seg: dict) -> str:
    if not seg["reasons"]:
        return "未知"
    name, _ = seg["reasons"].most_common(1)[0]
    return {"theta_deg": "颈线偏角", "head_drop": "低头",
            "head_pitch_deg": "头前倾", "shoulder_tilt": "肩歪"}.get(name, name)


def build_report(day: Optional[str] = None) -> Tuple[str, bool]:
    """返回 (报告文本, 是否有数据)。"""
    rows = _load(day)
    if not rows:
        return (f"{day or '今天'} 暂无数据。\n跑一天 app.py 后再来看报告。", False)

    counts = Counter(r["level"] for r in rows)
    total = len(rows)
    label = day or "今天"

    def pct(n: int) -> str:
        return f"{100.0 * n / total:5.1f}%"

    lines = [
        f"每日坐姿报告 —— {label}",
        f"记录 {total} 秒（约 {total / 3600:.1f} 小时）",
        f"  正常 {counts.get('good', 0):5d}s {pct(counts.get('good', 0))}"
        f"   注意 {counts.get('warn', 0):5d}s {pct(counts.get('warn', 0))}"
        f"   不良 {counts.get('bad', 0):5d}s {pct(counts.get('bad', 0))}"
        f"   未检测 {counts.get('unknown', 0):5d}s {pct(counts.get('unknown', 0))}",
        "",
        "时间线（每小时好坏分布，X=不良 w=注意 ·=正常 ?=未检测）",
    ]

    hours = _hour_ranges(rows)
    for h in hours:
        hr = [r for r in rows if _hour(r["timestamp"]) == h]
        if not hr:
            continue
        c = Counter(r["level"] for r in hr)
        n = len(hr)
        # 按秒数映射成 20 格条形
        seg = ""
        for lv in LEVELS:
            seg += LEVEL_MARK[lv] * max(0, round(20 * c.get(lv, 0) / n))
        seg = seg[:20].ljust(20, " ")
        bad_pct = 100.0 * (c.get("warn", 0) + c.get("bad", 0)) / n
        lines.append(f"  {h}:00 |{seg}| {bad_pct:5.1f}% 不良")

    lines += ["", "坏姿势 TOP 时段（连续 ≥3 秒，按时长排）"]
    segs = _bad_segments(rows)
    if not segs:
        lines.append("  今天没有持续的坏姿势，很好。")
    else:
        for i, s in enumerate(segs[:3], 1):
            n = sum(s["levels"].values())
            lines.append(f"  {i}. {s['start'][11:19]} ~ {s['end'][11:19]}"
                         f"  共 {n} 秒（不良 {s['levels'].get('bad', 0)}s"
                         f"/注意 {s['levels'].get('warn', 0)}s）  主因：{_main_reason(s)}")

    return "\n".join(lines), True


def save_report(day: Optional[str] = None, out_path: Optional[str] = None) -> str:
    """生成并写盘（data/report_YYYYMMDD.txt），返回路径。"""
    text, _ = build_report(day)
    out_path = out_path or os.path.join(storage.DATA_DIR,
                                        f"report_{day or __import__('datetime').datetime.now().strftime('%Y%m%d')}.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return out_path
