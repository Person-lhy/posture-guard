"""Day 13（M3-3）：姿态估计器对比实验 harness。

同一套下游指标（metrics.py），换不同姿态估计器，测三类指标：
  速度     平均/P50/P95 单帧延迟、FPS
  鲁棒     检出率（有效帧占比）
  稳定     角度一致性（theta/head_drop 标准差，越小越稳）

口径说明（写进报告，答辩照读）：
  角度一致性 ≠ 精度。没有人工标注真值时，只能测"输出稳不稳"，
  精度需要标注集做 PCK/OKS，超出本项目范围。
  两模型检出率不同（缺帧）时，角度稳定性不可直接对比。

用法（在装好依赖的机器上，沙箱里只有测试）：
    python benchmark.py                          # 摄像头 0，默认 120 帧
    python benchmark.py --source path/to/clip.mp4 --frames 200
    python benchmark.py --models mp,yolo         # 指定模型
    python benchmark.py --list                   # 只列可用模型

输出：data/benchmark_YYYYMMDD.txt（人读报告）+ .json（原始数据）
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import storage
from metrics import NOSE, LEFT_EAR, LEFT_SHOULDER, RIGHT_EAR, RIGHT_SHOULDER
from geometry import head_drop_ratio, midpoint, neck_tilt_deg, shoulder_tilt


# ---------------- 统计（纯函数，可测） ----------------


def _pct(sorted_vals: Sequence[float], p: float) -> float:
    """线性插值分位数，p ∈ [0,1]。输入须已排序。"""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


@dataclass
class Timing:
    latencies: List[float] = field(default_factory=list)   # 秒

    def add(self, dt: float) -> None:
        self.latencies.append(dt)

    def summary_ms(self) -> Dict[str, float]:
        if not self.latencies:
            return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "fps": 0.0}
        s = sorted(self.latencies)
        mean_s = statistics.fmean(s)
        return {
            "mean_ms": round(mean_s * 1000.0, 2),
            "p50_ms": round(_pct(s, 0.50) * 1000.0, 2),
            "p95_ms": round(_pct(s, 0.95) * 1000.0, 2),
            "fps": round(1.0 / mean_s, 1) if mean_s > 0 else 0.0,
        }


@dataclass
class BenchResult:
    model: str
    frames: int = 0
    detected: int = 0
    timing: Timing = field(default_factory=Timing)
    theta_samples: List[float] = field(default_factory=list)
    drop_samples: List[float] = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        return self.detected / self.frames if self.frames else 0.0

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "model": self.model,
            "frames": self.frames,
            "detected": self.detected,
            "detection_rate": round(self.detection_rate, 4),
        }
        out.update(self.timing.summary_ms())
        out["theta_std"] = round(statistics.pstdev(self.theta_samples), 3) \
            if len(self.theta_samples) > 1 else None
        out["drop_std"] = round(statistics.pstdev(self.drop_samples), 4) \
            if len(self.drop_samples) > 1 else None
        return out


# ---------------- 模型 runner（适配器层） ----------------


class PoseRunner(Protocol):
    name: str
    def detect(self, frame) -> Optional[Sequence]: ...


class MediaPipeRunner:
    name = "mediapipe"

    def __init__(self, complexity: int = 0):
        import cv2  # noqa: F401
        import mediapipe as mp
        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=complexity,
            smooth_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.name = f"mediapipe-c{complexity}"

    def detect(self, frame):
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._pose.process(rgb)
        return res.pose_landmarks.landmark if res.pose_landmarks else None


class YoloPoseRunner:
    name = "yolo-pose"

    def __init__(self, model_name: str = "yolo11n-pose.pt"):
        from ultralytics import YOLO
        self._yolo = YOLO(model_name)
        self.name = model_name.split(".")[0]

    def detect(self, frame):
        """返回 ultralytics keypoints（17 COCO 点，归一化到 0~1）。"""
        res = self._yolo.predict(frame, verbose=False)[0]
        if res.keypoints is None or res.keypoints.xy is None or len(res.keypoints.xy) == 0:
            return None
        import torch
        k = res.keypoints.xy[0]
        h, w = frame.shape[:2]
        return [(float(x) / w, float(y) / h, 0.0, 1.0) for x, y in k.cpu().numpy()]


def make_runners(spec: str = "mp,yolo") -> Tuple[List[Tuple[str, Callable]], List[str]]:
    """按 spec 构造 runner 工厂。返回 (可用 [(名字, 工厂)], 跳过原因列表)。

    工厂而非实例：bench 时逐模型构造，构造失败不影响后续模型。
    """
    made: List[Tuple[str, Callable]] = []
    skipped: List[str] = []
    for token in spec.split(","):
        t = token.strip().lower()
        if not t:
            continue
        if t == "mp":
            made.append(("MediaPipe Pose", lambda: MediaPipeRunner(0)))
        elif t == "mp1":
            made.append(("MediaPipe Pose (c1)", lambda: MediaPipeRunner(1)))
        elif t == "yolo":
            try:
                import ultralytics  # noqa: F401
                made.append(("YOLO11n-Pose", lambda: YoloPoseRunner("yolo11n-pose.pt")))
            except ImportError:
                skipped.append("YOLO11n-Pose：ultralytics 未安装（pip install ultralytics）")
        else:
            skipped.append(f"{t}：未知模型代号（可选 mp / mp1 / yolo）")
    return made, skipped


# ---------------- 帧读取（IO 层） ----------------


def load_frames(source: str, max_frames: int) -> List:
    """读帧到内存（比对要喂多个模型，生成器只能走一遍）。

    source："数字" = 摄像头号；视频文件路径；或图片目录。
    """
    import cv2

    frames: List = []
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {source}")
        try:
            while len(frames) < max_frames:
                ok, img = cap.read()
                if not ok:
                    break
                frames.append(img)
        finally:
            cap.release()
        return frames

    if os.path.isdir(source):
        names = sorted(n for n in os.listdir(source)
                       if n.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
        for n in names[:max_frames]:
            img = cv2.imread(os.path.join(source, n))
            if img is not None:
                frames.append(img)
        return frames

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频 {source}")
    try:
        while len(frames) < max_frames:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
    finally:
        cap.release()
    return frames


# ---------------- 测量（纯逻辑，clock 可注入） ----------------


def _angles(landmarks) -> Tuple[Optional[float], Optional[float]]:
    """从 landmarks 提取 (theta, drop)，失败返回 (None, None)。"""
    try:
        ear_mid = midpoint((landmarks[LEFT_EAR][0], landmarks[LEFT_EAR][1]),
                           (landmarks[RIGHT_EAR][0], landmarks[RIGHT_EAR][1]))
        sh_mid = midpoint((landmarks[LEFT_SHOULDER][0], landmarks[LEFT_SHOULDER][1]),
                          (landmarks[RIGHT_SHOULDER][0], landmarks[RIGHT_SHOULDER][1]))
        theta = neck_tilt_deg(ear_mid, sh_mid)
        scale = ((landmarks[LEFT_SHOULDER][0] - landmarks[RIGHT_SHOULDER][0]) ** 2 +
                 (landmarks[LEFT_SHOULDER][1] - landmarks[RIGHT_SHOULDER][1]) ** 2) ** 0.5
        drop = head_drop_ratio((landmarks[NOSE][0], landmarks[NOSE][1]), ear_mid, scale)
        return theta, drop
    except (IndexError, TypeError, ZeroDivisionError):
        return None, None


def bench(name: str, factory: Callable, frames: Sequence,
          clock: Callable[[], float] = time.perf_counter) -> BenchResult:
    """跑一个模型过全部帧，收集延迟/检出率/角度样本。"""
    result = BenchResult(model=name)
    try:
        runner = factory()
    except ImportError as e:
        result.model = f"{name}（依赖缺失：{e.name or e}）"
        return result

    for img in frames:
        t0 = clock()
        lms = runner.detect(img)
        result.timing.add(clock() - t0)
        result.frames += 1
        if lms is None:
            continue
        result.detected += 1
        theta, drop = _angles(lms)
        if theta is not None:
            result.theta_samples.append(theta)
        if drop is not None:
            result.drop_samples.append(drop)
    return result


# ---------------- 报告（纯函数） ----------------


def _fmt(v: Optional[float], n: int = 1, none: str = "–") -> str:
    return none if v is None else f"{v:.{n}f}"


def build_report(results: Sequence[BenchResult], skipped: Sequence[str] = (),
                 source: str = "", env: bool = True) -> str:
    lines: List[str] = ["姿态估计器对比实验报告", "=" * 46, ""]
    if env:
        try:
            import cv2
            cv = cv2.__version__
        except ImportError:
            cv = "未安装"
        lines += [
            f"环境：Python {platform.python_version()}  OpenCV {cv}  "
            f"{platform.system()} {platform.machine()}",
            f"数据源：{source or '未知'}    帧数：{results[0].frames if results else 0}",
            "",
        ]

    head = f"{'模型':<22}{'帧数':>6}{'检出率':>8}{'平均ms':>8}{'P50':>7}{'P95':>7}{'FPS':>7}{'theta_std':>11}{'drop_std':>10}"
    lines += [head, "-" * len(head)]
    for r in results:
        d = r.to_dict()
        rate = f"{d['detection_rate'] * 100:.1f}%" if d["frames"] else "–"
        lines.append(
            f"{d['model']:<22}{d['frames']:>6}{rate:>8}"
            f"{d['mean_ms']:>8.1f}{d['p50_ms']:>7.1f}{d['p95_ms']:>7.1f}{d['fps']:>7.1f}"
            f"{_fmt(d['theta_std'], 2):>11}{_fmt(d['drop_std'], 3):>10}")

    lines += ["", "口径说明："]
    lines += ["  · 检出率 = 有效帧占比；角度一致性 = theta/head_drop 标准差（越小越稳）。"]
    lines += ["  · 角度一致性 ≠ 精度：无标注真值，只测输出稳定性。"]
    lines += ["  · 两模型检出率差异大时（缺帧），角度一致性不可直接对比。"]
    if skipped:
        lines += ["", "跳过的模型："]
        lines += [f"  · {s}" for s in skipped]
    return "\n".join(lines)


def to_json(results: Sequence[BenchResult], skipped: Sequence[str] = ()) -> str:
    return json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "env": {"python": platform.python_version(), "system": platform.system()},
        "results": [r.to_dict() for r in results],
        "skipped": list(skipped),
    }, ensure_ascii=False, indent=2)


# ---------------- 入口 ----------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="姿态估计器对比实验")
    ap.add_argument("--source", default="0", help="摄像头号 / 视频文件 / 图片目录")
    ap.add_argument("--frames", type=int, default=120, help="最大帧数")
    ap.add_argument("--models", default="mp,yolo", help="模型代号，逗号分隔（mp / mp1 / yolo）")
    ap.add_argument("--list", action="store_true", help="只列可用模型，不跑")
    args = ap.parse_args(argv)

    made, skipped = make_runners(args.models)
    if args.list:
        print("可用模型：" + (", ".join(n for n, _ in made) or "无"))
        for s in skipped:
            print("跳过：" + s)
        return 0
    if not made:
        print("没有可用模型。" + ("；".join(skipped)))
        return 1

    print(f"读取帧（{args.source}，最多 {args.frames} 帧）…")
    try:
        frames = load_frames(args.source, args.frames)
    except RuntimeError as e:
        print(f"读帧失败：{e}")
        return 1
    if not frames:
        print("没读到任何帧")
        return 1

    results: List[BenchResult] = []
    for name, factory in made:
        print(f"跑 {name} …")
        results.append(bench(name, factory, frames))

    report = build_report(results, skipped, source=args.source)
    stamp = datetime.now().strftime("%Y%m%d")
    os.makedirs(storage.DATA_DIR, exist_ok=True)
    txt = os.path.join(storage.DATA_DIR, f"benchmark_{stamp}.txt")
    js = os.path.join(storage.DATA_DIR, f"benchmark_{stamp}.json")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(js, "w", encoding="utf-8") as f:
        f.write(to_json(results, skipped))
    print()
    print(report)
    print(f"\n已保存：{txt}\n原始数据：{js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
