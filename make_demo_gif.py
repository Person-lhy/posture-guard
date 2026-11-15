"""生成 README / 答辩用的示意演示动画 demo.gif。

合成数据驱动的原理演示（坐姿变化 → 指标越线 → 温和提醒 → 恢复），
**不是实机录屏**，画面上会标注这一点；实机录屏仍按 docs/答辩素材/演示视频脚本.md。

运行：python make_demo_gif.py        （需要 matplotlib + Pillow）
输出：docs/答辩素材/demo.gif
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 与 config.py 一致的真实阈值口径
TH_WARN, TH_RECOVER = 25.0, 18.0
D_WARN = 0.35

FPS = 12
N_FRAMES = 144  # 12 秒循环

C_BLUE = "#1F4E79"      # 深蓝（PPT 同款）
C_ACCENT = "#2E75B6"
C_GOOD = "#2E9E5B"
C_WARN = "#E6A23C"
C_BAD = "#D64541"
C_BG = "#FFFFFF"
C_CARD = "#F4F7FB"
C_TEXT = "#333333"
C_MUTED = "#7A8699"


def slouch(frame: int) -> float:
    """坐姿恶化程度 0（好）→ 1（差），分五段演绎。"""
    if frame < 24:
        return 0.0
    if frame < 60:
        return (frame - 24) / 36.0
    if frame < 84:
        return 1.0
    if frame < 120:
        return 1.0 - (frame - 84) / 36.0
    return 0.0


def metrics(s: float):
    """由恶化程度驱动的 3 项指标（与真实指标同名同量纲）。"""
    theta = 7.0 + 28.0 * s          # 颈偏角°：7 → 35
    drop = 0.12 + 0.32 * s          # 低头比 d：0.12 → 0.44
    tilt = 0.04 + 0.14 * s          # 肩差 s
    return theta, drop, tilt


class Hysteresis:
    """迟滞分级：θ ≥ 25 进警告，θ ≤ 18 才恢复（与 posture_rules 同口径）。"""

    def __init__(self):
        self.grade = "good"

    def update(self, theta: float) -> str:
        if self.grade == "good" and theta >= TH_WARN:
            self.grade = "bad"
        elif self.grade == "bad" and theta <= TH_RECOVER:
            self.grade = "good"
        elif self.grade == "good" and theta >= TH_WARN - 4:
            self.grade = "warn"
        elif self.grade == "warn" and theta <= TH_RECOVER:
            self.grade = "good"
        elif self.grade == "warn" and theta >= TH_WARN:
            self.grade = "bad"
        return self.grade


GRADE_COLOR = {"good": C_GOOD, "warn": C_WARN, "bad": C_BAD}
GRADE_TEXT = {"good": "良好坐姿", "warn": "注意坐姿", "bad": "不良坐姿"}


def build_figure():
    fig = plt.figure(figsize=(8.4, 5.2), dpi=100)
    fig.patch.set_facecolor(C_BG)

    # 顶部深蓝条
    fig.patches.append(Rectangle((0, 0.925), 1, 0.075, transform=fig.transFigure,
                                 facecolor=C_BLUE, edgecolor="none", zorder=0))
    fig.text(0.035, 0.958, "Posture Guard · 坐姿监测小助手", color="white",
             fontsize=14, fontweight="bold", va="center")
    fig.text(0.965, 0.958, "本地识别 · 温和提醒 · 数据不出本机", color="#BFD7EE",
             fontsize=9, va="center", ha="right")

    # 左：场景
    ax = fig.add_axes([0.035, 0.10, 0.53, 0.80])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("摄像头画面（示意）", fontsize=10, color=C_MUTED, pad=6)

    # 右：HUD
    hud = fig.add_axes([0.60, 0.10, 0.365, 0.80])
    hud.set_xlim(0, 10)
    hud.set_ylim(0, 10)
    hud.axis("off")

    # 底：时间线
    tl = fig.add_axes([0.075, 0.032, 0.89, 0.042])
    tl.set_xlim(0, N_FRAMES)
    tl.set_ylim(0, 1)
    tl.axis("off")

    fig.text(0.035, 0.053, "今日时间线", fontsize=7.5, color=C_MUTED, va="center")
    return fig, ax, hud, tl


def draw_scene(ax, s: float, bubble: bool):
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(0.15, 0.28, "示意动画 · 合成数据驱动，实测以本机运行结果为准",
            fontsize=6.5, color=C_MUTED)

    # 桌面与显示器
    ax.add_patch(Rectangle((5.6, 4.1), 3.9, 0.12, facecolor="#B9C7D6", edgecolor="none"))
    ax.add_patch(Rectangle((7.3, 4.22), 1.7, 2.6, facecolor="#2C3E50", edgecolor="none"))
    ax.add_patch(Rectangle((7.45, 4.35), 1.4, 2.3, facecolor="#8FB8DE", edgecolor="none"))
    ax.text(8.15, 6.95, "显示器", fontsize=7, color=C_MUTED, ha="center")

    # 椅子
    ax.add_patch(Rectangle((1.9, 2.0), 0.18, 1.6, facecolor="#9AA7B5", edgecolor="none"))
    ax.add_patch(Rectangle((1.35, 1.9), 1.5, 0.18, facecolor="#9AA7B5", edgecolor="none"))

    # 身体关键点（侧视，随 s 变形）
    hip = np.array([2.9, 3.55])
    shoulder = np.array([3.15 + 1.35 * s, 6.15 - 0.75 * s])
    head = np.array([3.30 + 2.0 * s, 7.35 - 1.55 * s])
    nose = head + np.array([0.62, -0.55 - 0.35 * s])
    ear = head + np.array([-0.30, 0.12])
    elbow = (shoulder + np.array([6.3, 4.25])) / 2 + np.array([0, -0.55 + 0.3 * s])
    hand = np.array([6.35, 4.32])

    lw = 3.2
    # 躯干 / 腿
    ax.plot([hip[0], shoulder[0]], [hip[1], shoulder[1]], color=C_ACCENT, lw=lw, solid_capstyle="round")
    ax.plot([hip[0], 2.35], [hip[1], 2.05], color=C_ACCENT, lw=lw, solid_capstyle="round")
    ax.plot([2.35, 3.9], [2.05, 2.05], color=C_ACCENT, lw=lw, solid_capstyle="round")
    # 手臂
    ax.plot([shoulder[0], elbow[0], hand[0]], [shoulder[1], elbow[1], hand[1]],
            color=C_ACCENT, lw=lw - 0.6, solid_capstyle="round")
    # 颈 + 头
    ax.plot([shoulder[0], head[0]], [shoulder[1], head[1]], color=C_ACCENT, lw=lw, solid_capstyle="round")
    ax.add_patch(plt.Circle(head, 0.52, facecolor="#DCE9F5", edgecolor=C_ACCENT, lw=2))
    ax.plot([head[0], nose[0]], [head[1], nose[1]], color=C_ACCENT, lw=2, solid_capstyle="round")
    ax.plot(*ear, "o", color=C_ACCENT, ms=4)
    ax.plot(*nose, "o", color=C_BAD if s > 0.6 else C_ACCENT, ms=4)

    # 肩线参考（虚线）提示肩差
    ax.plot([shoulder[0] - 1.1, shoulder[0] + 1.1], [shoulder[1], shoulder[1]],
            color=C_MUTED, lw=1, ls="--", alpha=0.6)

    if bubble:
        bb = FancyBboxPatch((4.6, 7.6), 4.6, 1.35, boxstyle="round,pad=0.18",
                            facecolor="#FFF4E0", edgecolor=C_WARN, lw=1.6)
        ax.add_patch(bb)
        ax.text(6.9, 8.55, "该坐直一下啦", fontsize=11, color="#8A5A00",
                ha="center", va="center", fontweight="bold")
        ax.text(6.9, 8.02, "坏姿势已持续一段时间 · 提醒一次即止", fontsize=7.5,
                color="#8A5A00", ha="center", va="center")
        ax.annotate("", xy=(head[0] + 0.55, head[1] + 0.35), xytext=(5.0, 7.9),
                    arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.4))


def draw_hud(hud, theta, drop, tilt, grade, status):
    hud.clear()
    hud.set_xlim(0, 10)
    hud.set_ylim(0, 10)
    hud.axis("off")

    hud.text(0.2, 9.55, "实时指标", fontsize=11, color=C_BLUE, fontweight="bold")

    rows = [
        ("θ 颈偏角", theta, 45.0, f"{theta:.1f}°"),
        ("d 低头比", drop, 0.60, f"{drop:.2f}"),
        ("s 肩差", tilt, 0.30, f"{tilt:.2f}"),
    ]
    y = 8.4
    for name, val, vmax, txt in rows:
        hud.text(0.2, y + 0.22, name, fontsize=8.5, color=C_TEXT)
        hud.add_patch(Rectangle((2.6, y - 0.12), 5.6, 0.5, facecolor="#E4EAF1",
                                edgecolor="none", zorder=1))
        frac = max(0.04, min(1.0, val / vmax))
        hud.add_patch(Rectangle((2.6, y - 0.12), 5.6 * frac, 0.5,
                                facecolor=GRADE_COLOR[grade], edgecolor="none", zorder=2))
        hud.text(9.7, y + 0.12, txt, fontsize=8.5, color=C_TEXT, ha="right")
        y -= 1.5

    # 判定 chip
    chip = FancyBboxPatch((1.5, 2.85), 7.0, 1.15, boxstyle="round,pad=0.16",
                          facecolor=GRADE_COLOR[grade], edgecolor="none")
    hud.add_patch(chip)
    hud.text(5.0, 3.42, f"判定：{GRADE_TEXT[grade]}", fontsize=12, color="white",
             ha="center", va="center", fontweight="bold")

    hud.text(5.0, 2.35, status, fontsize=8.5, color=C_ACCENT, ha="center")

    hud.text(0.2, 1.45, "提醒策略", fontsize=9, color=C_BLUE, fontweight="bold")
    hud.text(0.2, 0.92, "15s 持续触发 · 300s 冷却 · 600s 升级", fontsize=8, color=C_TEXT)
    hud.text(0.2, 0.42, "夜间静默 23:00–08:00 · 每日上限 20 次", fontsize=8, color=C_TEXT)


def draw_timeline(tl, grades):
    tl.clear()
    tl.set_xlim(0, N_FRAMES)
    tl.set_ylim(0, 1)
    tl.axis("off")
    tl.text(-1.0, 0.5, "时间线", fontsize=7.5, color=C_MUTED, va="center", ha="right")
    tl.add_patch(Rectangle((0, 0.18), N_FRAMES, 0.64, facecolor="#EDF1F6", edgecolor="none"))
    for i, g in enumerate(grades):
        tl.add_patch(Rectangle((i, 0.18), 1.05, 0.64, facecolor=GRADE_COLOR[g],
                               edgecolor="none"))


def status_for(frame: int, grade: str) -> str:
    if frame < 24:
        return "实时识别中 · 坐姿良好"
    if frame < 60:
        return "检测到坐姿变差 …"
    if frame < 72:
        return "持续异常 · 计时中（满 15s 才提醒）"
    if frame < 100:
        return "温和提醒一次 · 即将冷却"
    if frame < 120:
        return "恢复中 …"
    return "已恢复 · 保持就好"


def main() -> int:
    fig, ax, hud, tl = build_figure()
    hys = Hysteresis()
    grades: list = []

    def update(frame: int):
        s = slouch(frame)
        theta, drop, tilt = metrics(s)
        grade = hys.update(theta)
        grades.append(grade)
        bubble = 72 <= frame < 100
        draw_scene(ax, s, bubble)
        draw_hud(hud, theta, drop, tilt, grade, status_for(frame, grade))
        draw_timeline(tl, grades)

    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 // FPS)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "docs", "答辩素材", "demo.gif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
