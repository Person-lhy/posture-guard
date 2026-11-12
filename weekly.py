"""Day 12（M3-2）：周报 —— 近 7 天姿势趋势聚合 + 文本报告 + 趋势图。

数据源：data/ 下按天的 CSV（storage 写的那份）。聚合出每日不良率、
最长坏姿势段、各等级占比，文本报告可直接弹窗；趋势图走 matplotlib
（缺依赖时跳过，不影响文本报告）。

用法：
    weekly.build_text(days=7)          → 文本（纯 stdlib，可测）
    weekly.render_chart(days=7, path)  → PNG 路径（需 matplotlib）
    weekly.save_report()               → 文本+图都落盘，返回 (txt, png|None)
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import storage
from report import _bad_segments

LEVELS = ("good", "warn", "bad", "unknown")
LEVEL_CN = {"good": "正常", "warn": "注意", "bad": "不良", "unknown": "未检测"}


def _day_list(days: int, end: Optional[str] = None) -> List[str]:
    """近 N 天日期列表（YYYYMMDD，升序）。end 为结束日，不含则取今天。"""
    end_dt = datetime.strptime(end, "%Y%m%d") if end else datetime.now()
    return [(end_dt - timedelta(days=days - 1 - i)).strftime("%Y%m%d") for i in range(days)]


def _load_day(day: str) -> List[dict]:
    path = storage._csv_path(day)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate(days: int = 7, end: Optional[str] = None) -> List[Dict]:
    """近 N 天逐日聚合。无数据的天也会占位（records=0），保证时间轴连续。

    每项：{day, records, counts:{level: n}, bad_pct, longest_bad, reason}
      bad_pct    = (warn+bad)/records × 100
      longest_bad= 最长连续不良段秒数（0 = 无）
      reason     = 最长段主因（人读）
    """
    out: List[Dict] = []
    for day in _day_list(days, end):
        rows = _load_day(day)
        if not rows:
            out.append({"day": day, "records": 0, "counts": {lv: 0 for lv in LEVELS},
                        "bad_pct": 0.0, "longest_bad": 0, "reason": "-"})
            continue
        counts = {lv: sum(1 for r in rows if r["level"] == lv) for lv in LEVELS}
        n = len(rows)
        bad_pct = 100.0 * (counts["warn"] + counts["bad"]) / n if n else 0.0
        segs = _bad_segments(rows)
        longest = max((sum(s["levels"].values()) for s in segs), default=0)
        reason = "-"
        if segs:
            top = max(segs, key=lambda s: sum(s["levels"].values()))
            reasons = top.get("reasons")
            if reasons:
                name, _ = reasons.most_common(1)[0]
                reason = {"theta_deg": "颈线偏角", "head_drop": "低头",
                          "head_pitch_deg": "头前倾", "shoulder_tilt": "肩歪"}.get(name, name)
        out.append({"day": day, "records": n, "counts": counts,
                    "bad_pct": round(bad_pct, 1), "longest_bad": longest,
                    "reason": reason})
    return out


def build_text(days: int = 7, end: Optional[str] = None) -> Tuple[str, bool]:
    """文本周报。返回 (文本, 是否有数据)。"""
    rows = aggregate(days, end)
    if all(r["records"] == 0 for r in rows):
        return (f"近 {days} 天暂无数据。\n跑几天 app.py 后再来看周报。", False)

    start, stop = rows[0]["day"], rows[-1]["day"]
    fmt = lambda d: f"{d[4:6]}-{d[6:8]}"
    total = sum(r["records"] for r in rows)
    bad_days = [r for r in rows if r["bad_pct"] >= 20.0]

    lines = [
        f"周报 —— {fmt(start)} ~ {fmt(stop)}（近 {days} 天）",
        "",
        "日期    记录(s)   不良率   最长坏姿势      主因",
    ]
    for r in rows:
        if r["records"] == 0:
            lines.append(f"  {fmt(r['day'])}      -        -          -       -")
            continue
        mins = r["longest_bad"] // 60
        dur = f"{mins} 分 {r['longest_bad'] % 60:02d} 秒" if mins else f"{r['longest_bad']} 秒"
        lines.append(f"  {fmt(r['day'])}  {r['records']:6d}   {r['bad_pct']:5.1f}%   {dur:>10s}   {r['reason']}")

    avg_bad = sum(r["bad_pct"] for r in rows if r["records"]) / max(1, len(bad_days) or 1) \
        if any(r["records"] for r in rows) else 0.0
    worst = max((r for r in rows if r["records"]), key=lambda r: r["bad_pct"], default=None)
    lines += [
        "",
        f"合计记录 {total} 秒（约 {total / 3600:.1f} 小时）",
    ]
    if worst:
        lines.append(f"最差的一天：{fmt(worst['day'])}（不良率 {worst['bad_pct']}%，最长 {worst['longest_bad']} 秒）")
    if bad_days:
        lines.append(f"不良率 ≥20% 的天数：{len(bad_days)}/{sum(1 for r in rows if r['records'])}")
    else:
        lines.append("本周没有不良率 ≥20% 的日子，保持。")
    return "\n".join(lines), True


def render_chart(days: int = 7, out_path: Optional[str] = None,
                 end: Optional[str] = None) -> Optional[str]:
    """近 N 天趋势图（不良率折线 + 最长坏姿势柱），返回 PNG 路径。

    缺 matplotlib 时返回 None（文本报告不受影响）。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    rows = [r for r in aggregate(days, end)]
    labels = [f"{r['day'][4:6]}-{r['day'][6:8]}" for r in rows]
    bad = [r["bad_pct"] for r in rows]
    longest = [r["longest_bad"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(labels, longest, color="#f0997b", alpha=0.7, label="最长坏姿势(秒)")
    ax1.set_ylabel("最长坏姿势 (秒)")
    ax1.set_ylim(0, max(longest + [60]) * 1.2)

    ax2 = ax1.twinx()
    ax2.plot(labels, bad, "o-", color="#a32d2d", linewidth=2, label="不良率(%)")
    ax2.set_ylabel("不良率 (%)")
    ax2.set_ylim(0, 100)

    ax1.set_title("坐姿周报 —— 近 %d 天趋势" % days)
    ax1.grid(axis="y", alpha=0.3)
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    fig.tight_layout()

    out_path = out_path or os.path.join(storage.DATA_DIR, "weekly.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_report(days: int = 7) -> Tuple[str, Optional[str]]:
    """文本+图都落盘。返回 (文本路径, 图路径|None)。"""
    text, _ = build_text(days)
    stamp = datetime.now().strftime("%Y%m%d")
    os.makedirs(storage.DATA_DIR, exist_ok=True)
    txt_path = os.path.join(storage.DATA_DIR, f"weekly_{stamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    png_path = render_chart(days, os.path.join(storage.DATA_DIR, f"weekly_{stamp}.png"))
    return txt_path, png_path
