"""Day 8（M2-1）：用户设置持久化 + 开机自启。

config.py 的默认值是代码层；这里是用户层：改动写 JSON（data/settings.json），
启动时合并覆盖，不用改代码重启两遍。

开机自启走「启动文件夹 + .bat」路线（不用注册表）：
    %APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\Startup\\posture-guard.bat
好处：可见、可手动删、不需要管理员权限。非 Windows 平台 get_autostart 返回 False。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional

from config import REMINDERS, THRESHOLDS

SETTINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SETTINGS_PATH = os.path.join(SETTINGS_DIR, "settings.json")

# 可调字段：默认值取自 config.py
DEFAULTS: Dict[str, Any] = {
    "neck_tilt_warn": THRESHOLDS.neck_tilt_warn,
    "neck_tilt_recover": THRESHOLDS.neck_tilt_recover,
    "head_drop_warn": THRESHOLDS.head_drop_warn,
    "head_drop_recover": THRESHOLDS.head_drop_recover,
    "shoulder_tilt_warn": THRESHOLDS.shoulder_tilt_warn,
    "shoulder_tilt_recover": THRESHOLDS.shoulder_tilt_recover,
    "trigger_seconds": REMINDERS.trigger_seconds,
    "cooldown_seconds": REMINDERS.cooldown_seconds,
    "recover_seconds": REMINDERS.recover_seconds,
    "sedentary_seconds": REMINDERS.sedentary_seconds,
    "snooze_seconds": REMINDERS.snooze_seconds,
    "quiet_hours": list(REMINDERS.quiet_hours),
    "daily_alert_cap": REMINDERS.daily_alert_cap,
    "camera_index": 0,
}

_CURRENT: Dict[str, Any] = dict(DEFAULTS)


def load() -> Dict[str, Any]:
    """从 settings.json 读用户配置，缺失字段回退默认值。返回生效配置。"""
    global _CURRENT
    cfg = dict(DEFAULTS)
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, encoding="utf-8") as f:
                user = json.load(f)
            for k, v in user.items():
                if k in cfg:
                    cfg[k] = v
        except (OSError, ValueError):
            pass            # 配置损坏就用默认值，不让启动失败
    _CURRENT = cfg
    return cfg


def save(updates: Dict[str, Any]) -> Dict[str, Any]:
    """合并更新并写盘。只接受 DEFAULTS 里存在的键。返回生效配置。

    落盘前夹紧：recover 不得大于 warn（Grade 断言会崩），
    设置窗口手改和 autotune 自动写回都走这里，单点设防。
    """
    global _CURRENT
    cfg = dict(_CURRENT)
    for k, v in updates.items():
        if k in cfg:
            cfg[k] = v
    for prefix in ("neck_tilt", "head_drop", "shoulder_tilt"):
        w_key, r_key = f"{prefix}_warn", f"{prefix}_recover"
        if cfg.get(r_key, 0) > cfg.get(w_key, 0):
            cfg[r_key] = cfg[w_key]
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    _CURRENT = cfg
    return cfg


def current() -> Dict[str, Any]:
    return dict(_CURRENT)


def reset() -> Dict[str, Any]:
    """恢复默认值并清盘。"""
    global _CURRENT
    _CURRENT = dict(DEFAULTS)
    if os.path.exists(SETTINGS_PATH):
        try:
            os.remove(SETTINGS_PATH)
        except OSError:
            pass
    return dict(_CURRENT)


# ---------------- 开机自启（Windows 启动文件夹 + .bat） ----------------


def _bat_path() -> Optional[str]:
    if sys.platform != "win32":
        return None
    startup = os.path.join(
        os.environ.get("APPDATA", ""),
        "Microsoft", "Windows", "Start Menu", "Programs", "Startup")
    return os.path.join(startup, "posture-guard.bat")


def get_autostart() -> bool:
    p = _bat_path()
    return bool(p and os.path.exists(p))


def set_autostart(enable: bool) -> bool:
    """开关开机自启。返回操作后状态；失败（无权限/非 Windows）返回当前状态。"""
    p = _bat_path()
    if p is None:
        return False
    if enable:
        python = sys.executable
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
        content = (f'@echo off\r\n'
                   f'cd /d "{os.path.dirname(script)}"\r\n'
                   f'start "" "{python}" "{script}"\r\n')
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except OSError:
            return False
    try:
        os.remove(p)
        return False
    except OSError:
        return get_autostart()
