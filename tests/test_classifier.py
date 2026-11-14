"""classifier.py 单测：标准化数学、逻辑回归收敛、评估公式、序列化回环、数据加载。

运行：python tests/test_classifier.py   （或 python -m pytest tests/）
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import classifier  # noqa: E402
import storage  # noqa: E402


def _tmp_env():
    d = tempfile.mkdtemp(prefix="pg_clf_")
    old = storage.DATA_DIR
    storage.DATA_DIR = d
    return d, old


def test_standardize_math():
    X = [[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]]
    Z, means, stds = classifier.standardize(X)
    assert abs(means[0] - 3.0) < 1e-9 and abs(means[1] - 20.0) < 1e-9
    for c in (0, 1):
        col = [z[c] for z in Z]
        assert abs(sum(col) / len(col)) < 1e-9            # 0 均值
        var = sum(v * v for v in col) / len(col)
        assert abs(var - 1.0) < 1e-9                      # 1 方差


def test_standardize_constant_column_no_divzero():
    Z, means, stds = classifier.standardize([[2.0, 1.0], [2.0, 2.0]])
    assert stds[0] == 1.0                                 # 常数列防除零
    assert all(z[0] == 0.0 for z in Z)


def test_train_separable_converges():
    X, y = classifier.synthetic_dataset(200)
    Z, means, stds = classifier.standardize(X)
    w, b, losses = classifier.train_logistic(Z, y, lr=0.5, epochs=200)
    assert losses[-1] < losses[0]                         # 损失下降
    y_pred = [classifier.predict(w, b, x) for x in Z]
    acc = sum(1 for t, p in zip(y, y_pred) if t == p) / len(y)
    assert acc >= 0.95                                    # 可分数据应收敛


def test_predict_proba_range_and_boundary():
    w, b = [2.0, -1.0], 0.0
    p = classifier.predict_proba(w, b, [0.0, 0.0])
    assert abs(p - 0.5) < 1e-9                            # z=0 → 0.5
    assert 0.0 <= p <= 1.0
    assert classifier.predict(w, b, [0.0, 0.0], threshold=0.5) == 1   # 边界 p=0.5，>= 归 1
    assert classifier.predict(w, b, [0.0, 0.0], threshold=0.6) == 0
    assert classifier.predict(w, b, [5.0, 0.0]) == 1


def test_evaluate_metrics():
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 0, 1]
    m = classifier.evaluate(y_true, y_pred)
    assert abs(m["accuracy"] - 4 / 6) < 1e-3              # tp2 tn2（输出保留 4 位小数）
    assert abs(m["precision"] - 2 / 3) < 1e-3             # tp2 fp1
    assert abs(m["recall"] - 2 / 3) < 1e-3                # tp2 fn1
    f1 = 2 * (2 / 3) * (2 / 3) / (4 / 3)
    assert abs(m["f1"] - f1) < 1e-3


def test_save_load_roundtrip():
    d = tempfile.mkdtemp(prefix="pg_clf_model_")
    try:
        path = os.path.join(d, "m.json")
        classifier.save_model(path, [0.5, -0.25], 0.1, [1.0, 2.0], [0.5, 0.5])
        m = classifier.load_model(path)
        assert m["weights"] == [0.5, -0.25] and m["bias"] == 0.1
        assert m["means"] == [1.0, 2.0] and m["stds"] == [0.5, 0.5]
        assert m["features"] == list(classifier.FEATURES)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_load_dataset_from_csv():
    d, old = _tmp_env()
    try:
        storage.append_row("good", {"theta_deg": 3.0, "head_drop": 0.1,
                                    "head_pitch_deg": 0.0, "shoulder_tilt": 0.03,
                                    "neck_ratio": 0.5}, day="20260101")
        storage.append_row("bad", {"theta_deg": 30.0, "head_drop": 0.4,
                                   "head_pitch_deg": 0.0, "shoulder_tilt": 0.2,
                                   "neck_ratio": 0.4}, day="20260101")
        storage.append_row("warn", {"theta_deg": None, "head_drop": 0.4,
                                    "head_pitch_deg": 0.0, "shoulder_tilt": 0.2,
                                    "neck_ratio": 0.4}, day="20260101")
        X, y = classifier.load_dataset("20260101")
        assert len(X) == 2 and y == [0, 1]                # None 特征行被跳过
        assert X[0] == [3.0, 0.1, 0.03, 0.5]
    finally:
        storage.DATA_DIR = old
        shutil.rmtree(d, ignore_errors=True)


def test_run_experiment_end_to_end():
    X, y = classifier.synthetic_dataset(200)
    result = classifier.run_experiment(X, y)
    m = result["metrics"]
    assert m["train_samples"] == 140 and m["test_samples"] == 60
    assert m["accuracy"] >= 0.9
    assert 0.0 <= m["f1"] <= 1.0
    text = classifier.describe(result)
    assert "规则蒸馏" in text and "F1" in text


def test_split_holdout_sizes():
    X = [[float(i)] for i in range(10)]
    y = [i % 2 for i in range(10)]
    Xtr, Xte, ytr, yte = classifier.split_holdout(X, y)
    assert len(Xtr) == 7 and len(Xte) == 3 and len(ytr) == 7 and len(yte) == 3


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
