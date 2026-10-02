from config import Config, ensure_dirs


# ==========================================
# 报告摘要生成
# ==========================================
def build_report_summary(config=Config):
    ensure_dirs(config)
    lines = []
    lines.append("# AI 增强型量化选股与回测系统实验摘要\n\n")
    lines.append("## 数据说明\n")
    lines.append(f"- 标的: {config.stock_code}，数据来源: {config.data_source}。\n")
    lines.append(f"- 复权方式: {config.adjustment}。\n")
    lines.append("- 本项目不依赖新增 Tushare 下载，主要使用本地 OHLCV 与沪深 300 数据构造特征。\n\n")
    lines.append("## 模型说明\n")
    lines.append("- ARIMA/VAR 作为统计模型骨架，生成预测与残差特征。\n")
    lines.append("- ANN-LSTM 使用 20+ 维新特征重新训练，旧 10 维权重仅作为 legacy baseline 保存。\n\n")
    lines.append("## 输出文件\n")
    lines.append("- 表格输出位于 `outputs/tables`。\n")
    lines.append("- 图表输出位于 `outputs/figures`。\n")
    lines.append("- 新权重保存为 `checkpoints/exam_lstm_best.pth`。\n")
    path = config.report_dir / "exam_report_summary.md"
    path.write_text("".join(lines), encoding="utf-8")
    print(f"报告摘要已生成: {path}")
    return path
