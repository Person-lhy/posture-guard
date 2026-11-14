"""学习型坐姿分类器实验（离线，不进主链路）。

回答的问题：主链路的判定边界是「规则 + 分布标定」（posture_rules / autotune），
能不能直接从数据里学出判别边界？

方法：以现行规则分级为参照标签（good=0，warn/bad=1），取 4 项指标
（θ 颈偏角 / d 低头比值 / s 肩差 / r 颈长比）训练**纯 Python 逻辑回归**
（全批量梯度下降 + 特征标准化），留出集报告 准确率 / 精确率 / 召回率 / F1
以及与规则分级的一致率。体现完整训练流程：
数据加载 → 标准化 → 训练 → 评估 → 序列化，全程零第三方依赖。

口径（答辩照读）：这是**规则蒸馏**——标签来自现行规则，不是超越规则的
真值学习；价值在于 ① 训练管线的最小闭环证据 ② 学习边界与规则边界的量化对比。
若标签来自 autotune 校准后的个人阈值，则等价于"学习个人判别边界"。

用法：
    python classifier.py --demo            # 合成数据跑通全流程
    python classifier.py --day 20261115    # 用当日 CSV
    python classifier.py --day 20261115 --save   # 模型存到 data/model.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import storage

FEATURES = ("theta_deg", "head_drop", "shoulder_tilt", "neck_ratio")
MODEL_NAME = "model.json"


# ---------------- 数据 ----------------


def load_dataset(day: Optional[str] = None) -> Tuple[List[List[float]], List[int]]:
    """从当日 CSV 读 (X, y)。标签：good→0，warn/bad→1；特征缺失的行跳过。"""
    path = storage._csv_path(day)
    X: List[List[float]] = []
    y: List[int] = []
    if not os.path.exists(path):
        return X, y
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                feats = [float(row[name]) for name in FEATURES]
            except (KeyError, TypeError, ValueError):
                continue          # 特征缺失/损坏的行不进训练
            level = row.get("level", "")
            if level not in ("good", "warn", "bad"):
                continue
            X.append(feats)
            y.append(0 if level == "good" else 1)
    return X, y


def synthetic_dataset(n: int = 200) -> Tuple[List[List[float]], List[int]]:
    """合成可分数据（确定性，无随机种子问题）：good 低 θ/d，bad 高 θ/d。"""
    X: List[List[float]] = []
    y: List[int] = []
    for i in range(n):
        if i % 2 == 0:                       # good：小角度小低头
            X.append([2.0 + (i % 7) * 0.5, 0.10 + (i % 5) * 0.01,
                      0.02 + (i % 3) * 0.01, 0.52 + (i % 4) * 0.01])
            y.append(0)
        else:                                 # bad：大角度大低头
            X.append([26.0 + (i % 9) * 1.5, 0.38 + (i % 6) * 0.02,
                      0.16 + (i % 4) * 0.02, 0.40 + (i % 5) * 0.01])
            y.append(1)
    return X, y


# ---------------- 训练（纯 Python 逻辑回归） ----------------


def standardize(X: Sequence[Sequence[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    """按列标准化到 0 均值 1 方差；std=0 的列按 1 处理（防除零）。"""
    if not X:
        return [], [], []
    cols = len(X[0])
    means = [sum(r[c] for r in X) / len(X) for c in range(cols)]
    stds = []
    for c in range(cols):
        var = sum((r[c] - means[c]) ** 2 for r in X) / len(X)
        stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
    Z = [[(r[c] - means[c]) / stds[c] for c in range(cols)] for r in X]
    return Z, means, stds


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)                       # 负值分支防 exp 溢出
    return e / (1.0 + e)


def train_logistic(X: Sequence[Sequence[float]], y: Sequence[int],
                   lr: float = 0.5, epochs: int = 300) -> Tuple[List[float], float, List[float]]:
    """全批量梯度下降，返回 (weights, bias, loss曲线)。"""
    if not X:
        raise ValueError("空数据集无法训练")
    n, d = len(X), len(X[0])
    w = [0.0] * d
    b = 0.0
    losses: List[float] = []
    for _ in range(epochs):
        gw = [0.0] * d
        gb = 0.0
        loss = 0.0
        for i in range(n):
            z = sum(w[c] * X[i][c] for c in range(d)) + b
            p = _sigmoid(z)
            err = p - y[i]
            for c in range(d):
                gw[c] += err * X[i][c]
            gb += err
            # 二元交叉熵（数值稳定：夹紧 p）
            pc = min(max(p, 1e-9), 1.0 - 1e-9)
            loss += -(y[i] * math.log(pc) + (1 - y[i]) * math.log(1 - pc))
        for c in range(d):
            w[c] -= lr * gw[c] / n
        b -= lr * gb / n
        losses.append(loss / n)
    return w, b, losses


def predict_proba(w: Sequence[float], b: float, x: Sequence[float]) -> float:
    return _sigmoid(sum(w[c] * x[c] for c in range(len(w))) + b)


def predict(w: Sequence[float], b: float, x: Sequence[float], threshold: float = 0.5) -> int:
    return 1 if predict_proba(w, b, x) >= threshold else 0


# ---------------- 评估 ----------------


def evaluate(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    total = max(len(y_true), 1)
    acc = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": round(acc, 4), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4),
            "agreement_with_rules": round(acc, 4)}   # 标签来自规则，一致率=准确率


def split_holdout(X: Sequence[Sequence[float]], y: Sequence[int],
                  train_ratio: float = 0.7) -> Tuple[List, List, List, List]:
    """顺序切分（前 70% 训练、后 30% 留出）。确定性，可测。"""
    k = max(1, int(len(X) * train_ratio))
    return list(X[:k]), list(X[k:]), list(y[:k]), list(y[k:])


# ---------------- 序列化 ----------------


def save_model(path: str, w: Sequence[float], b: float,
               means: Sequence[float], stds: Sequence[float]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"weights": list(w), "bias": b,
                   "means": list(means), "stds": list(stds),
                   "features": list(FEATURES)}, f, ensure_ascii=False, indent=2)


def load_model(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------- 流程编排 ----------------


def run_experiment(rows_X: Sequence[Sequence[float]], rows_y: Sequence[int]) -> Dict[str, object]:
    """训练→评估全流程，返回指标 dict（demo 与真实数据共用）。"""
    if len(rows_X) < 20:
        raise ValueError(f"样本太少（{len(rows_X)} < 20），无法训练")
    Xtr, Xte, ytr, yte = split_holdout(rows_X, rows_y)
    Z, means, stds = standardize(Xtr)
    Zte = [[(Xte[i][c] - means[c]) / stds[c] for c in range(len(means))]
           for i in range(len(Xte))]
    w, b, losses = train_logistic(Z, ytr)
    y_pred = [predict(w, b, x) for x in Zte]
    metrics = evaluate(yte, y_pred)
    metrics["train_samples"] = len(Xtr)
    metrics["test_samples"] = len(Xte)
    metrics["final_loss"] = round(losses[-1], 4)
    return {"metrics": metrics, "weights": w, "bias": b,
            "means": means, "stds": stds}


def describe(result: Dict[str, object]) -> str:
    m = result["metrics"]
    lines = [
        "学习型分类器实验（规则蒸馏 · 纯 Python 逻辑回归）",
        "=" * 46,
        f"训练样本 {m['train_samples']}  留出样本 {m['test_samples']}",
        f"准确率 {m['accuracy']:.3f}  精确率 {m['precision']:.3f}  "
        f"召回率 {m['recall']:.3f}  F1 {m['f1']:.3f}",
        f"与规则分级一致率 {m['agreement_with_rules']:.3f}  末轮损失 {m['final_loss']:.4f}",
        "",
        "口径：标签来自现行规则分级（规则蒸馏），不是真值学习；",
        "      学习边界与规则边界的差异本身是观察对象。",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="学习型坐姿分类器实验")
    ap.add_argument("--demo", action="store_true", help="用合成数据跑通全流程")
    ap.add_argument("--day", default=None, help="用指定日期的 CSV（默认当天）")
    ap.add_argument("--save", action="store_true", help="模型存到 data/model.json")
    args = ap.parse_args(argv)

    if args.demo:
        X, y = synthetic_dataset()
        print(f"合成数据集：{len(X)} 样本（好坏各半）")
    else:
        X, y = load_dataset(args.day)
        print(f"当日数据：{len(X)} 样本（good={y.count(0)} warn/bad={y.count(1)}）")
    if len(X) < 20:
        print("样本不足 20 条，先跑一天再试（或用 --demo）")
        return 1

    result = run_experiment(X, y)
    print()
    print(describe(result))
    if args.save:
        path = os.path.join(storage.DATA_DIR, MODEL_NAME)
        save_model(path, result["weights"], result["bias"],
                   result["means"], result["stds"])
        print(f"\n模型已保存：{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
