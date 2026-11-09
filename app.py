"""Day 5：托盘常驻应用 —— 姿态监测 MVP 主程序。

链路：摄像头 → MediaPipe Pose → metrics → posture_rules（滤波/分级）→ reminder_engine（状态机）
      → 托盘三色图标 + 气泡提醒。

用法：
    python app.py           # 启动后驻留托盘，不弹窗口
    python app.py 1         # 指定摄像头设备号

托盘菜单：
    当前指标        —— 实时数值面板（每 1s 自刷新）
    小睡 30 分钟    —— 静默一段时间，到期自动恢复
    退出

线程模型（架构要点）：
    worker 线程：采集 + 推理 + 判定 + 状态机（CPU 密集，绝不能放 UI 线程）
    UI 线程：托盘、气泡、对话框
    通信：Qt signal（跨线程自动排队）→ UI 线程更新；用户动作经 engine.snooze() 走方法调用。
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict

import cv2
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QDoubleSpinBox,
                               QFormLayout, QHBoxLayout, QLabel, QMenu,
                               QMessageBox, QPushButton, QSpinBox,
                               QSystemTrayIcon, QVBoxLayout)

from config import REMINDERS, THRESHOLDS
from metrics import extract_metrics
from posture_rules import Grade, MedianFilter, analyze
from reminder_engine import ReminderEngine
from sedentary import SedentaryTracker
import autotune
import report
import settings
import storage
import weekly

LEVEL_COLORS = {"good": (80, 220, 80), "warn": (60, 200, 255),
                "bad": (60, 60, 230), "unknown": (170, 170, 170)}
LEVEL_NAMES = {"good": "正常", "warn": "注意", "bad": "不良", "unknown": "未检测"}


# ---------------------------------------------------------------- 工作线程


class Worker(QThread):
    """采集 + 推理 + 判定 + 状态机。所有 CPU 密集活都在这里。"""

    state_changed = Signal(dict)   # {"level","text","values"}
    alert_raised = Signal(dict)    # {"level","text","seconds"}

    def __init__(self, cam_index: int = 0):
        super().__init__()
        self.cam_index = cam_index
        self._running = True
        self._latest = {"level": "unknown", "text": "启动中…", "values": {}}
        self.engine = ReminderEngine()          # UI 线程可调 snooze()

    def stop(self) -> None:
        self._running = False

    def latest(self) -> dict:
        return self._latest

    def run(self) -> None:
        cfg = settings.load()
        cap = cv2.VideoCapture(cfg["camera_index"])
        if not cap.isOpened():
            self._latest = {"level": "unknown",
                            "text": f"无法打开摄像头 {cfg['camera_index']}", "values": {}}
            self.state_changed.emit(self._latest)
            return

        try:
            import mediapipe as mp
        except ImportError:
            self._latest = {"level": "unknown", "text": "缺少 mediapipe，pip install -r requirements.txt",
                            "values": {}}
            self.state_changed.emit(self._latest)
            cap.release()
            return

        mp_pose = mp.solutions.pose
        mf = MedianFilter(THRESHOLDS.median_window)
        grades = self._grades_from(cfg)        # 用户阈值（可热更，见 apply_settings）
        self.engine = ReminderEngine(
            trigger_seconds=cfg["trigger_seconds"],
            recover_seconds=cfg["recover_seconds"],
            cooldown_seconds=cfg["cooldown_seconds"],
            escalate_seconds=REMINDERS.escalate_seconds,
            snooze_seconds=cfg["snooze_seconds"])
        self._reminders = cfg
        self.sedentary = SedentaryTracker(
            threshold_seconds=cfg["sedentary_seconds"],
            cooldown_seconds=REMINDERS.snooze_seconds)   # 久坐冷却复用小睡时长量级
        last_emit = 0.0
        last_log = 0.0
        alerts_today = 0
        day = time.localtime().tm_yday

        with mp_pose.Pose(static_image_mode=False, model_complexity=0,
                          smooth_landmarks=True,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = pose.process(rgb)

                if result.pose_landmarks:
                    h, w = frame.shape[:2]
                    raw = extract_metrics(result.pose_landmarks.landmark, w, h)
                    ps = analyze(raw, mf, grades)
                else:
                    ps = analyze({"theta_deg": None, "head_drop": None,
                                  "head_pitch_deg": None, "neck_ratio": None,
                                  "shoulder_tilt": None, "confidence": 0.0}, mf, grades)

                now = time.monotonic()
                if now - last_emit >= 0.5:      # UI 刷新节流 2Hz，够用且省电
                    last_emit = now
                    self._latest = {"level": ps.level, "text": ps.text,
                                    "values": dict(ps.details)}
                    self.state_changed.emit(self._latest)
                if now - last_log >= 1.0:       # 数据落盘 1Hz，一天约 8 万行
                    last_log = now
                    try:
                        storage.append_row(ps.level, dict(ps.details),
                                           confidence=raw.get("confidence", 1.0)
                                           if result.pose_landmarks else 0.0)
                    except OSError:
                        pass                    # 磁盘满/权限问题不影响监测

                action = self.engine.update(ps)
                if action and action.get("type") == "alert":
                    today = time.localtime().tm_yday
                    if today != day:
                        day, alerts_today = today, 0
                    rem = self._reminders
                    if self._quiet_now(rem) or alerts_today >= rem["daily_alert_cap"]:
                        action["suppressed"] = True   # 静默时段/超上限：只记录不弹
                    else:
                        alerts_today += 1
                        self.alert_raised.emit(action)

                # 久坐线：独立于姿态线，同走提醒管线（静默/上限共用）
                sed = self.sedentary.update(ps.level)
                if sed:
                    today = time.localtime().tm_yday
                    if today != day:
                        day, alerts_today = today, 0
                    rem = self._reminders
                    if self._quiet_now(rem) or alerts_today >= rem["daily_alert_cap"]:
                        sed["suppressed"] = True
                    else:
                        alerts_today += 1
                        self.alert_raised.emit(sed)

        cap.release()

    @staticmethod
    def _grades_from(cfg: dict) -> dict:
        """按用户阈值构建分级器（与 posture_rules.default_grades 同构）。"""
        return {
            "theta_deg": Grade(cfg["neck_tilt_warn"], cfg["neck_tilt_recover"],
                               THRESHOLDS.neck_tilt_severe),
            "head_drop": Grade(cfg["head_drop_warn"], cfg["head_drop_recover"]),
            "head_pitch_deg": Grade(cfg["neck_tilt_warn"], cfg["neck_tilt_recover"],
                                    THRESHOLDS.neck_tilt_severe),
            "shoulder_tilt": Grade(cfg["shoulder_tilt_warn"], cfg["shoulder_tilt_recover"]),
        }

    def apply_settings(self, cfg: dict) -> None:
        """UI 线程保存设置后热更提醒参数（阈值下次启动生效，摄像头需重启）。"""
        self._reminders = cfg
        self.engine.trigger_seconds = cfg["trigger_seconds"]
        self.engine.recover_seconds = cfg["recover_seconds"]
        self.engine.cooldown_seconds = cfg["cooldown_seconds"]
        self.engine.snooze_seconds = cfg["snooze_seconds"]

    @staticmethod
    def _quiet_now(rem: dict) -> bool:
        now = time.strftime("%H:%M")
        start, end = rem["quiet_hours"]
        return start <= now or now < end if start > end else start <= now < end
        try:                                     # 退出时自动出当天校准报告
            storage.summary()
        except OSError:
            pass


# ---------------------------------------------------------------- 设置对话框


class SettingsDialog(QDialog):
    """阈值/提醒/摄像头/开机自启。保存后提醒参数热更，阈值与摄像头重启生效。"""

    def __init__(self, worker: "Worker", parent=None):
        super().__init__(parent)
        self.worker = worker
        self.setWindowTitle("坐姿监测 · 设置")
        self.setMinimumWidth(420)
        cfg = settings.current()

        form = QFormLayout()
        self.spins: dict = {}

        def spin(key, lo, hi, step=1, suffix="", decimals=0):
            if decimals:
                w = QDoubleSpinBox()
                w.setDecimals(decimals)
                w.setSingleStep(step)
            else:
                w = QSpinBox()
                w.setSingleStep(int(step))
            w.setRange(lo, hi)
            w.setValue(cfg[key])
            w.setSuffix(suffix)
            self.spins[key] = w
            return w

        form.addRow("<b>姿态阈值</b>", QLabel(""))
        form.addRow("颈线偏角警告", spin("neck_tilt_warn", 5, 60, 1, " °"))
        form.addRow("颈线偏角恢复", spin("neck_tilt_recover", 1, 59, 1, " °"))
        form.addRow("低头警告", spin("head_drop_warn", 0.05, 2.0, 0.01, "", 2))
        form.addRow("低头恢复", spin("head_drop_recover", 0.01, 1.99, 0.01, "", 2))
        form.addRow("肩歪警告", spin("shoulder_tilt_warn", 0.02, 1.0, 0.01, "", 2))
        form.addRow("肩歪恢复", spin("shoulder_tilt_recover", 0.01, 0.99, 0.01, "", 2))

        form.addRow("<b>提醒节奏</b>", QLabel(""))
        form.addRow("触发时长", spin("trigger_seconds", 5, 120, 1, " 秒"))
        form.addRow("恢复迟滞", spin("recover_seconds", 1, 60, 1, " 秒"))
        form.addRow("提醒冷却", spin("cooldown_seconds", 60, 1800, 10, " 秒"))
        form.addRow("久坐提醒", spin("sedentary_seconds", 300, 7200, 60, " 秒"))
        form.addRow("小睡时长", spin("snooze_seconds", 60, 7200, 60, " 秒"))
        form.addRow("每日上限", spin("daily_alert_cap", 1, 200, 1, " 次"))
        qh = cfg["quiet_hours"]
        self.quiet_start = QSpinBox(); self.quiet_start.setRange(0, 23)
        self.quiet_start.setValue(int(qh[0].split(":")[0]))
        self.quiet_end = QSpinBox(); self.quiet_end.setRange(0, 23)
        self.quiet_end.setValue(int(qh[1].split(":")[0]))
        row = QHBoxLayout()
        row.addWidget(self.quiet_start); row.addWidget(QLabel("时 至")); row.addWidget(self.quiet_end)
        row.addWidget(QLabel("时"))
        form.addRow("静默时段", row)

        form.addRow("<b>摄像头</b>", QLabel(""))
        self.cam_spin = spin("camera_index", 0, 8, 1)
        cam_row = QHBoxLayout()
        cam_row.addWidget(self.cam_spin)
        self.btn_test = QPushButton("测试")
        self.btn_test.clicked.connect(self.test_camera)
        cam_row.addWidget(self.btn_test)
        form.addRow("设备号", cam_row)

        self.chk_autostart = QCheckBox("开机自启（写入启动文件夹）")
        self.chk_autostart.setChecked(settings.get_autostart())
        form.addRow("", self.chk_autostart)

        btns = QHBoxLayout()
        btn_save = QPushButton("保存"); btn_save.setDefault(True)
        btn_save.clicked.connect(self.save)
        btn_cancel = QPushButton("关闭")
        btn_cancel.clicked.connect(self.reject)
        btns.addStretch(1); btns.addWidget(btn_save); btns.addWidget(btn_cancel)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addLayout(btns)

    def test_camera(self) -> None:
        idx = self.cam_spin.value()
        try:
            import cv2
        except ImportError:
            QMessageBox.warning(self, "测试", "缺少 opencv")
            return
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            QMessageBox.warning(self, "测试", f"无法打开摄像头 {idx}\n检查设备号或是否被占用")
            return
        ok, frame = cap.read()
        cap.release()
        if ok:
            h, w = frame.shape[:2]
            QMessageBox.information(self, "测试", f"摄像头 {idx} 可用，分辨率 {w}×{h}")
        else:
            QMessageBox.warning(self, "测试", f"摄像头 {idx} 打开成功但取不到画面")

    def save(self) -> None:
        updates = {k: w.value() for k, w in self.spins.items()}
        updates["quiet_hours"] = [f"{self.quiet_start.value():02d}:00",
                                  f"{self.quiet_end.value():02d}:00"]
        cfg = settings.save(updates)
        self.worker.apply_settings(cfg)          # 提醒参数热更
        want_auto = self.chk_autostart.isChecked()
        if want_auto != settings.get_autostart():
            settings.set_autostart(want_auto)
        note = "已保存。阈值与摄像头设备号重启后生效，提醒参数即时生效。"
        if settings.get_autostart() != want_auto:
            note += "\n开机自启设置失败（检查启动文件夹写权限）；当前状态：" + \
                    ("已启用" if settings.get_autostart() else "未启用")
        QMessageBox.information(self, "设置", note)
        self.accept()


# ---------------------------------------------------------------- 托盘 UI


class TrayApp:
    def __init__(self, cam_index: int):
        self.worker = Worker(cam_index)
        self.worker.state_changed.connect(self.on_state)
        self.worker.alert_raised.connect(self.on_alert)

        self.icon = QIcon(self._pixmap("unknown"))
        self.tray = QSystemTrayIcon(self.icon)
        self.tray.setToolTip("坐姿监测：启动中…")
        self.tray.activated.connect(self._on_activated)

        menu = QMenu()
        self.act_metrics = QAction("当前指标")
        self.act_metrics.triggered.connect(self.show_metrics)
        self.act_report = QAction("生成校准报告")
        self.act_report.triggered.connect(self.show_report)
        self.act_autotune = QAction("自动校准阈值")
        self.act_autotune.triggered.connect(self.run_autotune)
        self.act_daily = QAction("每日报告")
        self.act_daily.triggered.connect(self.show_daily)
        self.act_weekly = QAction("周报")
        self.act_weekly.triggered.connect(self.show_weekly)
        self.act_settings = QAction("设置…")
        self.act_settings.triggered.connect(self.show_settings)
        self.act_snooze = QAction(f"小睡 {int(REMINDERS.snooze_seconds // 60)} 分钟")
        self.act_snooze.triggered.connect(self.snooze)
        act_quit = QAction("退出")
        act_quit.triggered.connect(self.quit)
        menu.addAction(self.act_metrics)
        menu.addAction(self.act_daily)
        menu.addAction(self.act_weekly)
        menu.addAction(self.act_report)
        menu.addAction(self.act_autotune)
        menu.addAction(self.act_settings)
        menu.addSeparator()
        menu.addAction(self.act_snooze)
        menu.addSeparator()
        menu.addAction(act_quit)
        self.tray.setContextMenu(menu)

        self.dialog: QDialog | None = None
        self.label: QLabel | None = None
        self.report_dialog: QDialog | None = None
        self.settings_dialog: SettingsDialog | None = None
        self.daily_dialog: QDialog | None = None
        self.weekly_dialog: QDialog | None = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._refresh_dialog)
        self.timer.start(1000)

        self.tray.show()
        self.worker.start()

    # -- 图标 ------------------------------------------------------------

    @staticmethod
    def _pixmap(level: str, size: int = 64) -> QPixmap:
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        r, g, b = LEVEL_COLORS[level]
        p.setBrush(QColor(r, g, b))
        p.setPen(Qt.NoPen)
        p.drawEllipse(4, 4, size - 8, size - 8)
        p.end()
        return pm

    # -- 信号槽（UI 线程）------------------------------------------------

    def on_state(self, st: dict) -> None:
        level = st.get("level", "unknown")
        self.tray.setIcon(QIcon(self._pixmap(level)))
        self.tray.setToolTip(f"坐姿监测：{LEVEL_NAMES.get(level, level)}\n{st.get('text', '')}")
        self._refresh_dialog()

    def on_alert(self, act: dict) -> None:
        if act.get("type") == "sedentary":
            self.tray.showMessage("久坐提醒", act["text"],
                                  QSystemTrayIcon.Information, 10000)
        else:
            self.tray.showMessage("坐姿提醒", act["text"],
                                  QSystemTrayIcon.Warning, 8000)

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.Trigger:      # 左键单击
            self.show_metrics()

    # -- 菜单动作 --------------------------------------------------------

    def show_metrics(self) -> None:
        if self.dialog is None:
            self.dialog = QDialog()
            self.dialog.setWindowTitle("坐姿监测 · 当前指标")
            lay = QVBoxLayout(self.dialog)
            self.label = QLabel()
            self.label.setTextFormat(Qt.RichText)
            self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lay.addWidget(self.label)
        self._refresh_dialog()
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def _refresh_dialog(self) -> None:
        if self.label is None:
            return
        st = self.worker.latest()
        level = st.get("level", "unknown")
        r, g, b = LEVEL_COLORS[level]
        vals = st.get("values", {})
        rows = ""
        for k, v in vals.items():
            rows += f"<tr><td style='padding-right:18px'>{k}</td><td><b>{v}</b></td></tr>"
        self.label.setText(
            f"<h3 style='color:rgb({r},{g},{b})'>状态：{LEVEL_NAMES.get(level, level)}</h3>"
            f"<p style='color:gray'>{st.get('text', '')}</p>"
            f"<table>{rows}</table>"
            f"<p style='color:gray;font-size:11px'>阈值可在 config.py 调整；小睡/退出见托盘菜单。</p>")

    def snooze(self) -> None:
        self.worker.engine.snooze()
        mins = int(settings.current()["snooze_seconds"] // 60)
        self.tray.showMessage("坐姿监测",
                              f"已小睡 {mins} 分钟，期间只记录不提醒",
                              QSystemTrayIcon.Information, 4000)

    def show_settings(self) -> None:
        if self.settings_dialog is None:
            self.settings_dialog = SettingsDialog(self.worker)
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def run_autotune(self) -> None:
        """按当日真实分布自动校准阈值并写入设置（重启后生效）。"""
        sugg = autotune.suggest_from_day()
        text = autotune.describe(sugg)
        if not sugg:
            self.tray.showMessage("自动校准", text.split("\n")[0],
                                  QSystemTrayIcon.Information, 6000)
            return
        ret = QMessageBox.question(None, "自动校准阈值",
                                   text + "\n\n写入设置？",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ret == QMessageBox.Yes:
            autotune.apply(sugg)
            self.tray.showMessage("自动校准",
                                  "已写入设置，重启后生效", QSystemTrayIcon.Information, 5000)

    def show_daily(self) -> None:
        """每日报告：当日姿势时间线 + 坏姿势 TOP 时段。"""
        text, has = report.build_report()
        if not has:
            self.tray.showMessage("坐姿监测", text.split("\n")[0],
                                  QSystemTrayIcon.Information, 5000)
            return
        if self.daily_dialog is None:
            self.daily_dialog = QDialog()
            self.daily_dialog.setWindowTitle("坐姿监测 · 每日报告")
            lay = QVBoxLayout(self.daily_dialog)
            lbl = QLabel()
            lbl.setTextFormat(Qt.PlainText)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lbl.setObjectName("dailyLabel")
            lay.addWidget(lbl)
        lbl = self.daily_dialog.findChild(QLabel, "dailyLabel")
        lbl.setText(text)
        self.daily_dialog.show()
        self.daily_dialog.raise_()
        self.daily_dialog.activateWindow()

    def show_weekly(self) -> None:
        """周报：近 7 天趋势文本 + PNG 落盘（有 matplotlib 时）。"""
        text, has = weekly.build_text()
        if not has:
            self.tray.showMessage("坐姿监测", text.split("\n")[0],
                                  QSystemTrayIcon.Information, 5000)
            return
        txt_path, png_path = weekly.save_report()
        if self.weekly_dialog is None:
            self.weekly_dialog = QDialog()
            self.weekly_dialog.setWindowTitle("坐姿监测 · 周报")
            lay = QVBoxLayout(self.weekly_dialog)
            lbl = QLabel()
            lbl.setTextFormat(Qt.PlainText)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lbl.setObjectName("weeklyLabel")
            lay.addWidget(lbl)
        note = f"\n\n已保存：{txt_path}" + (f"\n趋势图：{png_path}" if png_path else "")
        self.weekly_dialog.findChild(QLabel, "weeklyLabel").setText(text + note)
        self.weekly_dialog.show()
        self.weekly_dialog.raise_()
        self.weekly_dialog.activateWindow()

    def show_report(self) -> None:
        """从当天 CSV 生成阈值校准报告并弹窗显示。"""
        try:
            path = storage.summary()
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            self.tray.showMessage("坐姿监测", f"生成报告失败：{e}",
                                  QSystemTrayIcon.Warning, 5000)
            return
        if self.report_dialog is None:
            self.report_dialog = QDialog()
            self.report_dialog.setWindowTitle("坐姿监测 · 阈值校准报告")
            lay = QVBoxLayout(self.report_dialog)
            lbl = QLabel()
            lbl.setTextFormat(Qt.PlainText)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lbl.setObjectName("reportLabel")
            lay.addWidget(lbl)
        lbl = self.report_dialog.findChild(QLabel, "reportLabel")
        lbl.setText(text)
        self.report_dialog.show()
        self.report_dialog.raise_()
        self.report_dialog.activateWindow()

    def quit(self) -> None:
        self.worker.stop()
        self.worker.wait(3000)
        self.tray.hide()
        QApplication.quit()


# ---------------------------------------------------------------- 入口


def main() -> int:
    cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("posture-guard")

    if not QSystemTrayIcon.isSystemTrayAvailable():
        print("系统托盘不可用，无法启动")
        return 1

    tray_app = TrayApp(cam)
    about_to_quit = app.aboutToQuit
    about_to_quit.connect(lambda: (tray_app.worker.stop(), tray_app.worker.wait(3000)))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
