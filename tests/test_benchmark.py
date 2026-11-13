"""benchmark.py 单测：分位数、Timing 汇总、BenchResult 字典、角度提取、
bench 测量循环（假 runner + 假时钟）、依赖缺失优雅跳过、报告格式、CLI 分发。

运行：python tests/test_benchmark.py   （或 python -m pytest tests/）
不依赖 cv2 / mediapipe / ultralytics：测量层与模型层分离，全部用假件钉死。
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmark  # noqa: E402


# ---------------- 假件 ----------------


class FakeClock:
    """每次调用前进 step 秒 → bench 里每帧延迟恒等于 step。"""

    def __init__(self, step: float = 0.02):
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        v = self.t
        self.t += self.step
        return v


class FakeRunner:
    """按预置序列返回 detect 结果的假姿态估计器。"""

    name = "fake"

    def __init__(self, results):
        self._results = list(results)
        self._i = 0

    def detect(self, img):
        r = self._results[self._i] if self._i < len(self._results) else None
        self._i += 1
        return r


def _lms(ear_l=(0.35, 0.3), ear_r=(0.75, 0.3), nose=(0.55, 0.36),
         sh_l=(0.4, 0.6), sh_r=(0.6, 0.6)):
    """13 个关键点的假 landmarks，锚点索引与 metrics.py 一致。"""
    pts = [(0.0, 0.0)] * 13
    pts[0] = nose
    pts[7], pts[8] = ear_l, ear_r
    pts[11], pts[12] = sh_l, sh_r
    return pts


# ---------------- 分位数与 Timing ----------------


def test_pct_edges():
    assert benchmark._pct([], 0.5) == 0.0
    assert benchmark._pct([5.0], 0.9) == 5.0
    assert benchmark._pct([1.0, 2.0, 3.0], 0.0) == 1.0
    assert benchmark._pct([1.0, 2.0, 3.0], 1.0) == 3.0
    assert abs(benchmark._pct([1.0, 2.0, 3.0, 4.0], 0.5) - 2.5) < 1e-9
    # p95 在 [0.02, 0.02, 0.02, 0.02, 0.10] 上：pos=3.8 → 0.02+0.08*0.8=0.084
    assert abs(benchmark._pct([0.02, 0.02, 0.02, 0.02, 0.10], 0.95) - 0.084) < 1e-9


def test_timing_summary():
    t = benchmark.Timing()
    assert t.summary_ms() == {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "fps": 0.0}

    t = benchmark.Timing()
    for v in (0.01, 0.02, 0.03):
        t.add(v)
    s = t.summary_ms()
    assert abs(s["mean_ms"] - 20.0) < 1e-6          # 平均 20ms
    assert abs(s["p50_ms"] - 20.0) < 1e-6
    assert abs(s["p95_ms"] - 29.0) < 1e-6           # 线性插值 0.02+0.01*0.9
    assert abs(s["fps"] - 50.0) < 1e-6


# ---------------- BenchResult ----------------


def test_bench_result_dict():
    r = benchmark.BenchResult(model="m")
    d = r.to_dict()
    assert d["frames"] == 0 and d["detected"] == 0
    assert d["detection_rate"] == 0.0
    assert d["theta_std"] is None and d["drop_std"] is None   # 样本 <2 不算标准差

    r = benchmark.BenchResult(model="m", frames=10, detected=8)
    r.theta_samples = [10.0, 10.0, 10.0]
    r.drop_samples = [0.30, 0.32]
    d = r.to_dict()
    assert d["detection_rate"] == 0.8
    assert d["theta_std"] == 0.0                              # 常数序列 pstdev=0
    assert abs(d["drop_std"] - 0.01) < 1e-9


# ---------------- 角度提取 ----------------


def test_angles_valid():
    theta, drop = benchmark._angles(_lms())
    # ear_mid=(0.55,0.3) sh_mid=(0.5,0.6) → v=(0.05,-0.3) → atan2(0.05,0.3)
    assert abs(theta - math.degrees(math.atan2(0.05, 0.3))) < 0.01
    # scale=肩距=0.2，drop=(0.36-0.3)/0.2=0.3
    assert abs(drop - 0.3) < 1e-9


def test_angles_bad_input():
    assert benchmark._angles([]) == (None, None)              # 点不够 → IndexError 路径
    assert benchmark._angles([(1.0, 2.0)] * 12) == (None, None)
    bad = _lms()
    bad[0] = None                                             # 锚点 None → TypeError 路径
    assert benchmark._angles(bad) == (None, None)
    flat = _lms(sh_l=(0.5, 0.6), sh_r=(0.5, 0.6))             # 肩宽 0 → scale 退化
    theta, drop = benchmark._angles(flat)
    assert drop is None                                       # head_drop_ratio 拒绝 scale≈0
    assert theta is not None


# ---------------- bench 测量循环 ----------------


def test_bench_counts_and_samples():
    frames = [object(), object(), object(), object()]
    seq = [_lms(), None, _lms(), _lms()]                      # 4 帧检出 3
    res = benchmark.bench("fake", lambda: FakeRunner(seq), frames, clock=FakeClock(0.02))
    assert res.model == "fake"
    assert res.frames == 4 and res.detected == 3
    assert abs(res.detection_rate - 0.75) < 1e-9
    assert len(res.timing.latencies) == 4                     # 未检出帧也计延迟
    s = res.timing.summary_ms()
    assert abs(s["mean_ms"] - 20.0) < 1e-6
    assert abs(s["fps"] - 50.0) < 1e-6
    assert len(res.theta_samples) == 3 and len(res.drop_samples) == 3


def test_bench_factory_import_error():
    def boom():
        raise ImportError("No module named 'ultralytics'", name="ultralytics")

    res = benchmark.bench("YOLO11n-Pose", boom, [object(), object()])
    assert "依赖缺失" in res.model and "ultralytics" in res.model
    assert res.frames == 0 and res.detected == 0
    assert res.timing.latencies == []


def test_bench_skips_angle_on_bad_landmarks():
    """detect 返回畸形 landmarks 时帧计数照走，只是不进角度样本。"""
    res = benchmark.bench("fake", lambda: FakeRunner([[None] * 13]),
                          [object()], clock=FakeClock())
    assert res.frames == 1 and res.detected == 1
    assert res.theta_samples == [] and res.drop_samples == []


# ---------------- make_runners 分发 ----------------


def test_make_runners_dispatch():
    made, skipped = benchmark.make_runners("mp")
    assert [n for n, _ in made] == ["MediaPipe Pose"]
    assert skipped == []

    made, skipped = benchmark.make_runners("mp1")
    assert [n for n, _ in made] == ["MediaPipe Pose (c1)"]

    made, skipped = benchmark.make_runners("bogus")
    assert made == []
    assert len(skipped) == 1 and "未知模型代号" in skipped[0]

    made, skipped = benchmark.make_runners("mp,,  yolo ,")
    assert len(made) + len(skipped) == 2                     # 空 token 被忽略


def test_make_runners_yolo_missing_or_present():
    """yolo 依赖可装可不装：要么进 made，要么带原因跳过——两种都算对。"""
    made, skipped = benchmark.make_runners("yolo")
    names = [n for n, _ in made]
    assert any("YOLO" in n for n in names) or any("ultralytics" in s for s in skipped)


# ---------------- 报告 ----------------


def _filled_result():
    r = benchmark.BenchResult(model="MediaPipe Pose", frames=10, detected=8)
    r.timing.latencies = [0.01] * 8 + [0.02] * 2
    r.theta_samples = [10.0, 10.0, 10.0]
    r.drop_samples = [0.30, 0.32]
    return r


def test_build_report_format():
    txt = benchmark.build_report([_filled_result()],
                                 skipped=["YOLO11n-Pose：ultralytics 未安装"],
                                 source="0", env=False)
    assert "姿态估计器对比实验报告" in txt
    assert "MediaPipe Pose" in txt
    assert "口径说明" in txt
    assert "角度一致性 ≠ 精度" in txt                  # 答辩口径必须在报告里
    assert "检出率差异大" in txt
    assert "跳过的模型" in txt and "ultralytics 未安装" in txt
    assert "80.0%" in txt                              # 检出率百分数

    txt2 = benchmark.build_report([], env=True)       # 空结果 + 环境行，不崩
    assert "数据源" in txt2


def test_to_json_roundtrip():
    j = benchmark.to_json([_filled_result()], skipped=["x"])
    data = json.loads(j)
    assert data["results"][0]["model"] == "MediaPipe Pose"
    assert data["results"][0]["detection_rate"] == 0.8
    assert data["skipped"] == ["x"]
    assert "generated" in data and "env" in data


# ---------------- CLI ----------------


def test_main_list_and_no_models():
    assert benchmark.main(["--list", "--models", "mp"]) == 0
    assert benchmark.main(["--models", "bogus"]) == 1        # 没有可用模型


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
