"""把最近一次 benchmark 结果同步进 README 的实验结果表。

用法（跑完对比实验后执行一次）：
    python benchmark.py && python sync_bench.py

逻辑：找 data/benchmark_*.json 里日期最新的一份 → 渲染表格 →
替换 README.md 中 <!-- BENCH:START --> 与 <!-- BENCH:END --> 之间的内容。
"""

from __future__ import annotations

import glob
import os
import re
import sys
from typing import Dict, List, Optional

import storage

START = "<!-- BENCH:START -->"
END = "<!-- BENCH:END -->"


def render_table(data: Dict) -> str:
    """benchmark.to_json 的 dict → markdown 表格（含口径一行）。"""
    rows = data.get("results", [])
    lines = [
        "| 模型 | 帧数 | 检出率 | 平均ms | P50 | P95 | FPS | theta_std | drop_std |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rate = f"{r.get('detection_rate', 0) * 100:.1f}%" if r.get("frames") else "–"
        lines.append(
            f"| {r.get('model', '–')} | {r.get('frames', 0)} | {rate} "
            f"| {r.get('mean_ms', 0):.1f} | {r.get('p50_ms', 0):.1f} | {r.get('p95_ms', 0):.1f} "
            f"| {r.get('fps', 0):.1f} | {r.get('theta_std') if r.get('theta_std') is not None else '–'} "
            f"| {r.get('drop_std') if r.get('drop_std') is not None else '–'} |")
    skipped = data.get("skipped") or []
    if skipped:
        lines.append("")
        lines.append("跳过：" + "；".join(skipped))
    lines.append("")
    lines.append(f"实测时间：{data.get('generated', '–')}　口径：角度一致性 ≠ 精度（无标注真值只测输出稳定性）")
    return "\n".join(lines)


def update_readme(readme_text: str, table: str) -> str:
    """替换标记段内容；标记缺失时原样返回。"""
    if START not in readme_text or END not in readme_text:
        return readme_text
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    return pattern.sub(START + "\n" + table + "\n" + END, readme_text, count=1)


def latest_benchmark() -> Optional[str]:
    paths = glob.glob(os.path.join(storage.DATA_DIR, "benchmark_*.json"))
    return max(paths) if paths else None    # 文件名含 YYYYMMDD，字典序即时间序


def main() -> int:
    path = latest_benchmark()
    if not path:
        print("data/benchmark_*.json 不存在——先跑 python benchmark.py")
        return 1
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    table = render_table(data)

    readme = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
    with open(readme, "r", encoding="utf-8") as f:
        text = f.read()
    new_text = update_readme(text, table)
    if new_text == text:
        print("README 里没有 BENCH 标记段，未改动")
        return 1
    with open(readme, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"已同步 {os.path.basename(path)} → README.md 实验结果表")
    return 0


if __name__ == "__main__":
    sys.exit(main())
