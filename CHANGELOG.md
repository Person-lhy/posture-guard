# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 风格；版本号语义化（SemVer）。

## [1.0.0] - 2026-11-15

首个完整版本：从采集到提醒、从日报到校准的本地闭环。

### Added

- **实时识别**：MediaPipe Pose 33 关键点 → θ（颈偏角）/ d（低头比值）/ r（颈长比）/ φ（俯仰角）/ s（肩差）5 项几何指标；5 帧中值滤波 + 迟滞分级（warn 进线 / recover 出线分离）
- **智能提醒**：状态机 NORMAL → COUNTDOWN → ALERTED → SNOOZED；15 s 触发、300 s 冷却、600 s 升级、夜间静默 23:00–08:00、每日上限 20 次、小睡 30 min；判定（posture_rules）与提醒（reminder_engine）分层
- **久坐提醒**：坐满 45 min 提示起身，离座自动清零
- **数据闭环**：每秒指标落盘 CSV → 每日报告（24 h 时间线 + 坏姿势 TOP 时段）→ 周报双轴趋势图 → 阈值校准（warn ≈ 不良 P15 / recover ≈ 正常 P85，分布重叠时拒绝给建议）→ 一键写回配置
- **学习型分类器实验**：规则蒸馏口径的纯 Python 逻辑回归（4 特征，标准留出评估：准确率 / P / R / F1 / 与规则一致率）
- **对比实验 harness**：MediaPipe vs YOLO-Pose 测速 / 检出率 / 稳定性，结果一键同步 README 表格（`sync_bench.py`）
- **工程配套**：PySide6 托盘 + 设置窗口、开机自启（启动文件夹，不碰注册表）、108 个单元测试（假时钟 + 合成关键点，零重依赖可回归）、GitHub Actions CI（3.10 / 3.11 × Ubuntu / Windows）

### Notes

- 口径诚实：基准表在实测前不填数字；一致性不等于精度，精度需标注集支撑（见 Roadmap）

[1.0.0]: https://github.com/Person-lhy/posture-guard/releases/tag/v1.0
