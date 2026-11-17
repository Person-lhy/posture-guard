"""precision_eval.py 单测：PCK/角度误差的数学、I/O 校验、报告口径、演示对端到端。

运行：python tests/test_precision_eval.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from precision_eval import (  # noqa: E402
    KEYPOINT_NAMES,
    Sample,
    build_report,
    keypoint_norm_errors,
    load_samples,
    make_detector,
    pck_at,
    shoulder_width,
    summarize,
    task_errors,
    task_metrics,
    write_demo_files,
)


def _kp(nose=(100.0, 90.0), le=(80.0, 86.0), re=(120.0, 88.0),
        ls=(70.0, 200.0), rs=(130.0, 202.0)):
    return {"nose": nose, "left_ear": le, "right_ear": re,
            "left_shoulder": ls, "right_shoulder": rs}


def test_shoulder_width_normal():
    w = shoulder_width(_kp())
    # sqrt(60^2 + 2^2)
    assert abs(w - 60.0333) < 1e-3


def test_shoulder_width_degenerate_returns_none():
    kp = _kp(ls=(100.0, 200.0), rs=(100.0, 200.0))
    assert shoulder_width(kp) is None


def test_task_metrics_upright_theta_near_zero():
    # 耳中 (100,87) 正在肩中 (100,201) 上方 → θ=0；鼻低于耳 → d>0
    m = task_metrics(_kp())
    assert abs(m["theta_deg"]) < 1e-6
    assert m["head_drop"] > 0


def test_keypoint_norm_errors_perfect_pred():
    gt = _kp()
    errs = keypoint_norm_errors(gt, dict(gt))
    assert all(v == 0.0 for v in errs.values())
    assert set(errs) == set(KEYPOINT_NAMES)


def test_keypoint_norm_errors_offset_one_scale():
    gt = _kp()
    w = shoulder_width(gt)
    pred = {k: (v[0] + w, v[1]) for k, v in gt.items()}
    errs = keypoint_norm_errors(gt, pred)
    for v in errs.values():
        assert abs(v - 1.0) < 1e-9


def test_pck_at_threshold_boundary_inclusive():
    gt = _kp()
    w = shoulder_width(gt)
    pred = {k: (v[0] + 0.1 * w, v[1]) for k, v in gt.items()}  # 误差恰 0.1
    errs = [keypoint_norm_errors(gt, pred)]
    overall, per = pck_at(errs, 0.1)
    assert overall == 1.0
    assert all(h == 1.0 for h in per.values())
    overall2, _ = pck_at(errs, 0.05)
    assert overall2 == 0.0


def test_pck_at_per_point_partial():
    gt = _kp()
    pred = dict(gt)
    pred["nose"] = (gt["nose"][0] + 1000, gt["nose"][1])  # 只有鼻偏出
    errs = [keypoint_norm_errors(gt, pred)]
    overall, per = pck_at(errs, 0.1)
    assert per["nose"] == 0.0
    assert per["left_ear"] == 1.0
    assert abs(overall - 4 / 5) < 1e-9


def test_task_errors_identical_zero():
    gt = _kp()
    errs = task_errors(gt, dict(gt))
    assert errs["theta_deg"] == 0.0
    assert errs["head_drop"] == 0.0
    assert errs["shoulder_tilt"] == 0.0


def test_task_errors_known_theta():
    # 真值竖直；预测耳中整体右移 10px（肩中不动）→ Δθ = atan2(10,114)≈5.01°
    gt = _kp()
    pred = dict(gt)
    pred["left_ear"] = (gt["left_ear"][0] + 10, gt["left_ear"][1])
    pred["right_ear"] = (gt["right_ear"][0] + 10, gt["right_ear"][1])
    errs = task_errors(gt, pred)
    assert abs(errs["theta_deg"] - 5.0136) < 1e-3


def test_summarize_math():
    st = summarize([1.0, 2.0, 3.0, 4.0])
    assert st["n"] == 4
    assert abs(st["mae"] - 2.5) < 1e-9
    assert abs(st["rmse"] - (30 / 4) ** 0.5) < 1e-9
    assert abs(st["max"] - 4.0) < 1e-9
    assert summarize([])["n"] == 0


def test_load_samples_rejects_missing_keypoint(tmp=None):
    d = tempfile.mkdtemp()
    try:
        bad = {"samples": [{"id": "x", "keypoints": {"nose": [1, 2]}}]}
        p = os.path.join(d, "bad.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(bad, f)
        try:
            load_samples(p)
            assert False, "应抛 ValueError"
        except ValueError as e:
            assert "缺关键点" in str(e)
    finally:
        shutil.rmtree(d)


def test_load_samples_pred_priority_and_fallback():
    d = tempfile.mkdtemp()
    try:
        gt = _kp()
        pred_in = {k: (v[0] + 3, v[1]) for k, v in gt.items()}
        pred_out = {k: (v[0] + 9, v[1]) for k, v in gt.items()}
        ann = {"samples": [
            {"id": "a", "keypoints": gt, "pred": pred_in},   # 内嵌 pred 优先
            {"id": "b", "keypoints": gt},                    # 走 pred_path
            {"id": "c", "keypoints": gt},                    # 无预测
        ]}
        pr = {"samples": [{"id": "b", "keypoints": pred_out}]}
        pa = os.path.join(d, "ann.json")
        pp = os.path.join(d, "pred.json")
        with open(pa, "w", encoding="utf-8") as f:
            json.dump(ann, f)
        with open(pp, "w", encoding="utf-8") as f:
            json.dump(pr, f)
        out = load_samples(pa, pp)
        assert out[0].pred["nose"] == pred_in["nose"]
        assert out[1].pred["nose"] == pred_out["nose"]
        assert out[2].pred is None
    finally:
        shutil.rmtree(d)


def test_build_report_refuses_without_pred():
    s = Sample(id="s1", gt=_kp())
    rep = build_report([s])
    assert "拒绝出数字" in rep
    assert "s1" in rep


def test_build_report_demo_flag_and_pck():
    d = tempfile.mkdtemp()
    try:
        ann, pred = write_demo_files(d)
        samples = load_samples(ann, pred)
        rep = build_report(samples, thresh=0.1, demo=True)
        assert "演示模式" in rep
        assert "PCK@0.1" in rep
        assert "口径" in rep
        # 演示对：6 个命中、2 个偏出 → 整体 75%
        assert "75.0%" in rep
    finally:
        shutil.rmtree(d)


def test_write_demo_files_roundtrip():
    d = tempfile.mkdtemp()
    try:
        ann, pred = write_demo_files(d)
        assert os.path.exists(ann) and os.path.exists(pred)
        samples = load_samples(ann, pred)
        assert len(samples) == 8
        assert all(s.pred is not None for s in samples)
    finally:
        shutil.rmtree(d)


def test_make_detector_missing_deps_returns_reason():
    # 沙箱无 mediapipe/cv2 时返回 (None, 原因)；有依赖的机器返回可调用
    detect, reason = make_detector()
    if detect is None:
        assert "mediapipe" in reason
    else:
        assert callable(detect)


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
