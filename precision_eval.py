"""Day 22（M3-4）：精度评测 harness —— PCK + 任务角度误差（需人工标注真值）。

与 benchmark.py 的分工：
    benchmark.py      测"输出稳不稳"（角度一致性，无真值，可自动化）
    precision_eval.py 测"输出准不准"（精度，必须人工标注，本模块只算不采）

口径（写进报告，答辩照读）：
    PCK@τ          预测关键点落在真值 τ×肩宽 阈值内的比例（定位精度，τ 默认 0.1）
    θ / d / s 误差 任务指标绝对误差的 MAE / RMSE / P95（直接对下游判定负责）
    没有标注文件就拒绝输出数字；--demo 生成的合成对只演示流程，报告标"演示"。

标注格式见 docs/标注规范.md（5 个锚点：鼻 / 左耳 / 右耳 / 左肩 / 右肩，像素坐标 y 向下）。

用法：
    python precision_eval.py --ann data/annotations.json
    python precision_eval.py --ann data/annotations.json --pred data/predictions.json
    python precision_eval.py --ann data/annotations.json --images data/frames/
    python precision_eval.py --demo            # 生成演示用合成标注/预测对并出报告
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from geometry import head_drop_ratio, midpoint, neck_tilt_deg, shoulder_tilt

Point = Tuple[float, float]
KEYPOINT_NAMES = ("nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder")


# ---------------- 小工具（纯函数） ----------------


def _pct(vals: Sequence[float], p: float) -> float:
    """线性插值分位数，p∈[0,1]。输入须已排序。"""
    if not vals:
        return 0.0
    if len(vals) == 1:
        return float(vals[0])
    pos = (len(vals) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return float(vals[lo] + (vals[hi] - vals[lo]) * frac)


def shoulder_width(kp: Dict[str, Point]) -> Optional[float]:
    """归一化尺度：双肩像素距。退化（双肩重合）返回 None。"""
    dx = kp["left_shoulder"][0] - kp["right_shoulder"][0]
    dy = kp["left_shoulder"][1] - kp["right_shoulder"][1]
    w = math.hypot(dx, dy)
    return w if w > 1e-6 else None


def task_metrics(kp: Dict[str, Point]) -> Dict[str, Optional[float]]:
    """5 锚点 → 任务指标 θ/d/s（与 metrics.extract_metrics 同几何，不含 φ/r）。"""
    ear_mid = midpoint(kp["left_ear"], kp["right_ear"])
    sh_mid = midpoint(kp["left_shoulder"], kp["right_shoulder"])
    scale = shoulder_width(kp)
    return {
        "theta_deg": neck_tilt_deg(ear_mid, sh_mid),
        "head_drop": head_drop_ratio(kp["nose"], ear_mid, scale) if scale else None,
        "shoulder_tilt": shoulder_tilt(kp["left_shoulder"], kp["right_shoulder"]),
    }


# ---------------- 误差与汇总（纯函数） ----------------


def keypoint_norm_errors(gt: Dict[str, Point], pred: Dict[str, Point]) -> Dict[str, float]:
    """逐锚点欧氏距离 / 真值肩宽（无量纲）。缺尺度时抛 ValueError。"""
    scale = shoulder_width(gt)
    if scale is None:
        raise ValueError("真值双肩重合，无法归一化")
    out: Dict[str, float] = {}
    for name in KEYPOINT_NAMES:
        dx = pred[name][0] - gt[name][0]
        dy = pred[name][1] - gt[name][1]
        out[name] = math.hypot(dx, dy) / scale
    return out


def pck_at(norm_errors: Sequence[Dict[str, float]], thresh: float) -> Tuple[float, Dict[str, float]]:
    """PCK@thresh：全部锚点命中率 + 逐锚点命中率。命中 = 误差 ≤ thresh（含边界）。"""
    if not norm_errors:
        return 0.0, {n: 0.0 for n in KEYPOINT_NAMES}
    per: Dict[str, float] = {}
    for name in KEYPOINT_NAMES:
        hits = sum(1 for e in norm_errors if e[name] <= thresh)
        per[name] = hits / len(norm_errors)
    overall = sum(per.values()) / len(KEYPOINT_NAMES)
    return overall, per


def task_errors(gt: Dict[str, Point], pred: Dict[str, Point]) -> Dict[str, Optional[float]]:
    """任务指标绝对误差：|pred − gt|，任一侧算不出则该指标为 None。"""
    mg, mp = task_metrics(gt), task_metrics(pred)
    out: Dict[str, Optional[float]] = {}
    for k in ("theta_deg", "head_drop", "shoulder_tilt"):
        out[k] = None if (mg[k] is None or mp[k] is None) else abs(mp[k] - mg[k])
    return out


def summarize(vals: Sequence[float]) -> Dict[str, float]:
    """误差序列 → n/MAE/RMSE/P95/max。"""
    vals = [float(v) for v in vals]
    if not vals:
        return {"n": 0, "mae": 0.0, "rmse": 0.0, "p95": 0.0, "max": 0.0}
    srt = sorted(vals)
    return {
        "n": len(vals),
        "mae": statistics.fmean(vals),
        "rmse": math.sqrt(statistics.fmean([v * v for v in vals])),
        "p95": _pct(srt, 0.95),
        "max": srt[-1],
    }


# ---------------- 样本 I/O ----------------


@dataclass
class Sample:
    id: str
    gt: Dict[str, Point]
    pred: Optional[Dict[str, Point]] = None


def _parse_kp(raw: object, sid: str) -> Dict[str, Point]:
    if not isinstance(raw, dict):
        raise ValueError(f"样本 {sid}: keypoints 应为对象")
    kp: Dict[str, Point] = {}
    for name in KEYPOINT_NAMES:
        if name not in raw:
            raise ValueError(f"样本 {sid}: 缺关键点 {name}")
        xy = raw[name]
        if not (isinstance(xy, (list, tuple)) and len(xy) >= 2):
            raise ValueError(f"样本 {sid}: {name} 坐标应为 [x, y]")
        kp[name] = (float(xy[0]), float(xy[1]))
    return kp


def load_samples(ann_path: str, pred_path: Optional[str] = None) -> List[Sample]:
    """读标注（必填 gt）与预测（可选独立文件 / 样本内 pred 字段）。

    预测来源优先级：样本内 pred 字段 > pred_path 同 id 行。都缺则 pred=None（出报告时拒绝）。
    """
    with open(ann_path, encoding="utf-8") as f:
        data = json.load(f)
    raw_samples = data.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("标注文件无 samples 列表")

    preds: Dict[str, Dict[str, Point]] = {}
    if pred_path:
        with open(pred_path, encoding="utf-8") as f:
            pdata = json.load(f)
        for item in pdata.get("samples", []):
            preds[str(item["id"])] = _parse_kp(item.get("keypoints"), str(item["id"]))

    out: List[Sample] = []
    for item in raw_samples:
        sid = str(item.get("id", len(out)))
        gt = _parse_kp(item.get("keypoints"), sid)
        pred = None
        if isinstance(item.get("pred"), dict):
            pred = _parse_kp(item["pred"], sid)
        elif sid in preds:
            pred = preds[sid]
        out.append(Sample(id=sid, gt=gt, pred=pred))
    return out


# ---------------- 检测器适配器（懒加载，缺依赖优雅降级） ----------------


def make_detector() -> Tuple[Optional[Callable[[str], Dict[str, Point]]], str]:
    """返回 (检测函数或 None, 跳过原因)。检测函数：图片路径 → 5 锚点像素坐标。"""
    try:
        import cv2  # noqa: F401
        import mediapipe as mp  # noqa: F401
    except ImportError:
        return None, "mediapipe/cv2 未安装，无法从图片生成预测（pip install mediapipe opencv-python）"

    def detect(image_path: str) -> Dict[str, Point]:
        import cv2
        import mediapipe as mp

        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"无法读取图片 {image_path}")
        h, w = img.shape[:2]
        with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
            res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            raise RuntimeError(f"未检出人体 {image_path}")
        lms = res.pose_landmarks.landmark
        idx = {"nose": 0, "left_ear": 7, "right_ear": 8,
               "left_shoulder": 11, "right_shoulder": 12}
        return {n: (lms[i].x * w, lms[i].y * h) for n, i in idx.items()}

    return detect, ""


# ---------------- 报告 ----------------


def build_report(samples: List[Sample], thresh: float = 0.1, demo: bool = False) -> str:
    """人读报告。样本缺预测 → 拒绝出数字，如实列出缺哪些。"""
    missing = [s.id for s in samples if s.pred is None]
    lines: List[str] = ["# 精度评测报告", ""]
    if demo:
        lines += ["> **演示模式**：样本为合成数据，数字只演示流程，不是实测精度。", ""]
    if missing:
        lines += [f"**拒绝出数字**：{len(missing)}/{len(samples)} 个样本没有预测值"
                  f"（{', '.join(missing[:5])}…）。",
                  "补 --pred 文件或 --images 走检测器后再跑。", ""]
        return "\n".join(lines)

    norm_errors = [keypoint_norm_errors(s.gt, s.pred) for s in samples]
    overall, per = pck_at(norm_errors, thresh)
    lines += [f"样本数：{len(samples)}　PCK@{thresh}（误差≤{thresh:.0%} 肩宽）：", "",
              f"- **整体命中率 {overall:.1%}**",
              *[f"- {n}: {per[n]:.1%}" for n in KEYPOINT_NAMES], ""]

    lines += ["## 任务指标误差（|pred − gt|）", "",
              "| 指标 | n | MAE | RMSE | P95 | max |", "|---|---|---|---|---|---|"]
    units = {"theta_deg": "°", "head_drop": "", "shoulder_tilt": ""}
    for key, unit in units.items():
        errs = [task_errors(s.gt, s.pred)[key] for s in samples]
        st = summarize([e for e in errs if e is not None])
        lines.append(f"| {key}{('（' + unit + '）') if unit else ''} | {st['n']} | "
                     f"{st['mae']:.3f} | {st['rmse']:.3f} | {st['p95']:.3f} | {st['max']:.3f} |")

    lines += ["",
              "口径：PCK 是关键点定位精度（阈值按肩宽归一化）；θ/d/s 误差是任务层指标，"
              "直接对应下游分级阈值的量纲。样本量 <30 时汇总统计仅供个案参考。"]
    return "\n".join(lines)


# ---------------- 演示数据（只用于 --demo 与测试） ----------------


def write_demo_files(out_dir: str) -> Tuple[str, str]:
    """生成合成 gt/pred 对（确定性），演示标注格式与评测流程。返回 (ann_path, pred_path)。"""
    os.makedirs(out_dir, exist_ok=True)
    gt_samples, pr_samples = [], []
    # 逐点不同的缩放系数：平移不改变相对几何，抖动必须逐点不同才能让 θ/d/s 误差可见
    mult = {"nose": (1.2, 1.0), "left_ear": (0.9, 0.8), "right_ear": (1.0, 0.9),
            "left_shoulder": (0.6, 0.7), "right_shoulder": (0.7, 0.85)}
    for i in range(8):
        x0, y0 = 200.0 + 5 * i, 150.0 + 3 * i
        gt = {
            "nose": [x0 + 2, y0 + 12], "left_ear": [x0 - 30, y0 + 8], "right_ear": [x0 + 34, y0 + 9],
            "left_shoulder": [x0 - 40, y0 + 120], "right_shoulder": [x0 + 42, y0 + 122],
        }
        # 前 6 个小抖动（PCK@0.1 命中），后 2 个大偏移（不命中）
        base = 2.0 * (i % 3)
        j = 0.0 if i < 6 else 20.0
        pred = {k: [v[0] + (base + j) * mult[k][0], v[1] + (base + j) * mult[k][1]]
                for k, v in gt.items()}
        gt_samples.append({"id": f"demo{i}", "keypoints": gt})
        pr_samples.append({"id": f"demo{i}", "keypoints": pred})

    ann_path = os.path.join(out_dir, "demo_annotations.json")
    pred_path = os.path.join(out_dir, "demo_predictions.json")
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "coord": "pixel, y down", "samples": gt_samples}, f,
                  ensure_ascii=False, indent=1)
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "coord": "pixel, y down", "samples": pr_samples}, f,
                  ensure_ascii=False, indent=1)
    return ann_path, pred_path


# ---------------- CLI ----------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="精度评测（PCK + 任务角度误差，需标注真值）")
    ap.add_argument("--ann", help="标注 JSON 路径")
    ap.add_argument("--pred", help="预测 JSON 路径（或样本内 pred 字段）")
    ap.add_argument("--images", help="图片目录：用 MediaPipe 现场生成预测")
    ap.add_argument("--thresh", type=float, default=0.1, help="PCK 归一化阈值（默认 0.1 肩宽）")
    ap.add_argument("--out", help="报告输出路径（缺省打印）")
    ap.add_argument("--demo", action="store_true", help="生成合成演示对并出报告（演示口径）")
    args = ap.parse_args(argv)

    demo = bool(args.demo)
    if demo:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "precision_demo")
        ann, pred = write_demo_files(out_dir)
        print(f"演示标注/预测已生成：{ann} / {pred}")
    else:
        if not args.ann:
            ap.error("需要 --ann 标注文件（或 --demo 看流程）")
        ann, pred = args.ann, args.pred

    samples = load_samples(ann, pred)

    if args.images:
        detect, reason = make_detector()
        if detect is None:
            print(f"跳过检测器：{reason}")
            if any(s.pred is None for s in samples):
                print("样本无内嵌预测，无法出报告。")
                return 1
        else:
            for s in samples:
                if s.pred is None:
                    img = os.path.join(args.images, f"{s.id}.jpg")
                    s.pred = detect(img)

    report = build_report(samples, thresh=args.thresh, demo=demo)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"报告已写入：{args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
