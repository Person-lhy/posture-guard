"""Day 11（M3-1）：阈值自适应 —— 按当日真实分布自动校准阈值。

校准报告（storage.summary）给出建议值，但要人眼对照着抄进设置。
autotune 把这一步自动化：读当日 CSV → 算建议阈值 → 可直接写入 settings.json。

建议规则（与校准报告同一套，见 storage.summary）：
  warn    ≈ 不良姿势下沿（bad P15）——低于它几乎不可能是坏姿势
  recover ≈ 正常姿势上沿（good P85）——回到它以下才算真恢复
  两分布重叠（bad P15 ≤ good P85）时该指标不给建议，硬给会把正常姿势误报成坏姿势。

样本不足（< MIN_SAMPLES）只给单侧建议：有好姿势样本才给 recover，有坏姿势样本才给 warn。
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

import settings
import storage
from storage import _quantile

MIN_SAMPLES = 20

# CSV 指标 → settings 阈值前缀。head_pitch_deg 是实验性 z 指标，不参与自动校准。
METRICS = {
    "theta_deg": "neck_tilt",
    "head_drop": "head_drop",
    "shoulder_tilt": "shoulder_tilt",
}


def _load_rows(day: Optional[str] = None) -> List[dict]:
    path = storage._csv_path(day)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _values(rows: List[dict], name: str, level: str) -> List[float]:
    out: List[float] = []
    for r in rows:
        if r.get("level") != level:
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


def compute_suggestions(rows: List[dict], min_samples: int = MIN_SAMPLES) -> Dict[str, float]:
    """从 CSV 行算出可直接写入 settings 的阈值建议。

    返回形如 {"neck_tilt_warn": 28.4, "neck_tilt_recover": 6.7, ...}。
    数据不足或无区分度的指标直接跳过——宁缺毋滥，绝不给会误报的阈值。
    """
    out: Dict[str, float] = {}
    for name, prefix in METRICS.items():
        good = _values(rows, name, "good")
        bad = _values(rows, name, "bad")
        g85 = _quantile(good, 0.85) if len(good) >= min_samples else None
        b15 = _quantile(bad, 0.15) if len(bad) >= min_samples else None

        if g85 is not None and b15 is not None:
            if b15 <= g85:
                continue                        # 分布重叠，无区分度
            out[f"{prefix}_warn"] = round(b15, 2)
            out[f"{prefix}_recover"] = round(g85, 2)
        elif b15 is not None:
            out[f"{prefix}_warn"] = round(b15, 2)      # 只有坏姿势样本
        elif g85 is not None:
            out[f"{prefix}_recover"] = round(g85, 2)   # 只有好姿势样本
    return out


def suggest_from_day(day: Optional[str] = None) -> Dict[str, float]:
    """从当日（或指定日）CSV 算建议阈值。"""
    return compute_suggestions(_load_rows(day))


def apply(suggestions: Dict[str, float]) -> Dict[str, object]:
    """写入 settings.json（recover>warn 由 settings.save 落盘前夹紧）。返回生效配置。"""
    return settings.save(suggestions)


def describe(suggestions: Dict[str, float]) -> str:
    """人读文案，用于托盘弹窗展示建议内容。"""
    if not suggestions:
        return "今日数据不足或分布无区分度，没有可自动校准的阈值。继续收集数据后再试。"
    names = {"neck_tilt_warn": "颈线偏角警告", "neck_tilt_recover": "颈线偏角恢复",
             "head_drop_warn": "低头警告", "head_drop_recover": "低头恢复",
             "shoulder_tilt_warn": "肩歪警告", "shoulder_tilt_recover": "肩歪恢复"}
    lines = [f"{names.get(k, k)} → {v}" for k, v in sorted(suggestions.items())]
    return "按今日数据建议：\n" + "\n".join(lines) + "\n\n（重启后生效）"
