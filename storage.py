"""Day 6：实测数据记录与阈值校准报告。

把每帧判级结果落盘 CSV（一天一个文件），再从真实分布反推阈值建议
——这是"拍脑袋定阈值"换"实测定阈值"的那一步，答辩里的标定故事就从这来。

CSV 列（与 posture_rules / metrics 的字段名一致）：
    timestamp, level, confidence,
    theta_deg, head_drop, head_pitch_deg, shoulder_tilt, neck_ratio

用法：
    storage.append_row(...)          # 每秒一次，app.py 的 worker 在写
    storage.summary()                # 生成 data/summary_YYYYMMDD.txt 并返回路径
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from config import THRESHOLDS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_COLUMNS = ["timestamp", "level", "confidence",
               "theta_deg", "head_drop", "head_pitch_deg", "shoulder_tilt", "neck_ratio"]


def _quantile(values: Sequence[float], q: float) -> float:
    """线性插值分位数（纯标准库，替代 pandas）。q ∈ [0, 1]。"""
    if not values:
        raise ValueError("empty")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _csv_path(day: Optional[str] = None) -> str:
    return os.path.join(DATA_DIR, f"{day or datetime.now().strftime('%Y%m%d')}.csv")


def append_row(level: str, values: Dict[str, Optional[float]],
               confidence: float = 1.0,
               timestamp: Optional[str] = None,
               day: Optional[str] = None) -> str:
    """追加一行到当天 CSV，返回文件路径。values 用 metrics 的字段名。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = _csv_path(day)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    row = {
        "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
        "level": level,
        "confidence": confidence,
        "theta_deg": values.get("theta_deg"),
        "head_drop": values.get("head_drop"),
        "head_pitch_deg": values.get("head_pitch_deg"),
        "shoulder_tilt": values.get("shoulder_tilt"),
        "neck_ratio": values.get("neck_ratio"),
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return path


def summary(day: Optional[str] = None, out_path: Optional[str] = None) -> str:
    """按当天 CSV 的真实分布生成阈值校准报告，返回报告路径。

    思路（细节规格 §1.4-4 的"自适应阈值"雏形）：
      - warn 阈值 ≈ 正常姿势下该指标 85 分位 × 1.2（留余量）
      - recover 阈值 ≈ 不良姿势下该指标 15 分位 × 0.9（留余量）
      - θ / φ 带符号，取绝对值后统计（判级用的就是 |v|）
    """
    path = _csv_path(day)
    out_path = out_path or os.path.join(DATA_DIR, f"summary_{day or datetime.now().strftime('%Y%m%d')}.txt")
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(path):
        text = f"无数据文件：{path}\n先跑 app.py 收集一段真实数据再生成报告。"
    else:
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            text = f"数据文件为空：{path}"
        else:
            def col(name: str, levels: Optional[set] = None) -> List[float]:
                out: List[float] = []
                for r in rows:
                    if levels is not None and r["level"] not in levels:
                        continue
                    v = r.get(name, "")
                    if v in ("", None):
                        continue
                    try:
                        x = float(v)
                    except ValueError:
                        continue
                    out.append(abs(x) if name in ("theta_deg", "head_pitch_deg") else x)
                return out

            n_good = sum(1 for r in rows if r["level"] == "good")
            n_warn = sum(1 for r in rows if r["level"] == "warn")
            n_bad = sum(1 for r in rows if r["level"] == "bad")
            conf = col("confidence")
            lines = [
                f"坐姿监测阈值校准报告 —— {day or datetime.now().strftime('%Y-%m-%d')}",
                f"样本数 {len(rows)}  好 {n_good}  警告 {n_warn}  不良 {n_bad}",
                f"置信度均值 {sum(conf) / len(conf):.2f}" if conf else "置信度均值 n/a",
                "",
                "指标         正常P50  正常P85  不良P50  不良P15  当前warn  当前recover",
            ]
            recs: List[str] = []
            for name, label in [("theta_deg", "theta"), ("head_drop", "drop"),
                                ("head_pitch_deg", "pitch"), ("shoulder_tilt", "shldr")]:
                good = col(name, {"good"})
                bad = col(name, {"bad"})
                if not good and not bad:
                    continue
                g50 = _quantile(good, 0.5) if good else float("nan")
                g85 = _quantile(good, 0.85) if good else float("nan")
                b50 = _quantile(bad, 0.5) if bad else float("nan")
                b15 = _quantile(bad, 0.15) if bad else float("nan")
                if name == "theta_deg":
                    cw, cr = THRESHOLDS.neck_tilt_warn, THRESHOLDS.neck_tilt_recover
                elif name == "head_drop":
                    cw, cr = THRESHOLDS.head_drop_warn, THRESHOLDS.head_drop_recover
                elif name == "head_pitch_deg":
                    cw, cr = THRESHOLDS.neck_tilt_warn, THRESHOLDS.neck_tilt_recover
                else:
                    cw, cr = THRESHOLDS.shoulder_tilt_warn, THRESHOLDS.shoulder_tilt_recover
                lines.append(f"{label:<12} {g50:8.3f} {g85:8.3f} {b50:8.3f} {b15:8.3f}"
                             f"  {cw:9.2f}  {cr:11.2f}")
                if len(good) >= 20 and len(bad) >= 20:
                    # 建议阈值锚定分布边界，不是正常分布尾部：
                    #   warn    ≈ 不良姿势下沿（bad P15）——低于它就几乎不可能是坏姿势
                    #   recover ≈ 正常姿势上沿（good P85）——回到它以下才算真恢复
                    # 无区分度（两分布重叠：bad P15 ≤ good P85）时不给建议，
                    # 硬给会把正常姿势误报成坏姿势
                    if b15 > g85:
                        recs.append(f"  {label} warn     → {b15:.2f}  (不良P15，分布下沿)")
                        recs.append(f"  {label} recover  → {g85:.2f}  (正常P85，分布上沿)")
                    else:
                        recs.append(f"  {label}：正常/不良分布重叠，无区分度，不建议单独用此指标")
            lines += ["", "建议阈值（样本≥20 才给出）:"]
            lines += recs or ["  样本不足，继续收集"]
            lines += ["", "填回 config.py Thresholds 后重启生效。"]
            text = "\n".join(lines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return out_path
