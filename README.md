# ANN-LSTM：A 股收益率预测与量化策略研究

一套围绕单只 A 股标的（`300502.SZ` 新易盛）的**完整量化研究流程**：从数据清洗与平稳性检验出发，横向对比传统统计模型（ARIMA / VAR）与深度学习模型（并行 ANN-LSTM），再通过集成策略融合两者信号，最后落到一套可回测的 A 股交易策略上。

项目重点不在于堆模型，而在于把**统计检验 → 建模 → 集成 → 回测**这条链路完整走通，并保证全程无未来数据泄漏。

> 课程项目 / 研究性质，结论仅供学习参考，不构成任何投资建议。

## 研究流程

项目按 6 个阶段推进，`outputs/` 下的产物以阶段号 `L1`–`L6` 命名：

| 阶段 | 内容 | 主要产出 |
|---|---|---|
| L1 | 数据清洗与描述性统计 | `outputs/tables/L1_*` |
| L2 | 平稳性检验（ADF/KPSS）、日历效应分析 | `L2_stationarity_tests.csv`、`L2_acf_pacf.png` |
| L3 | ARIMA 建模与滚动预测、ARCH 效应检验、Ljung-Box 白噪声检验 | `L3_arima_*.csv`、`L3_arch_effect.csv` |
| L4 | 向量自回归（VAR）、Granger 因果检验、Johansen 协整、脉冲响应分析 | `L4_var_irf.png`、`L4_granger_tests.csv` |
| L5 | 并行 ANN-LSTM 深度学习预测 | `train_exam_lstm.py` |
| L6 | 统计模型与深度模型的集成预测 + 策略回测 | `L6_backtest_equity.png` |

## 项目结构

```
.
├── config.py               # 全局配置（路径、超参、股票代码）
├── data_loader.py          # 数据读取与清洗
├── feature_engineering.py  # 20+ 维技术指标特征构造
├── feature_selection.py    # 特征筛选（相关性 / 重要性）
├── statistical_models.py   # ARIMA / VAR / Granger / Johansen 等统计模型
├── train_exam_lstm.py      # 并行 ANN-LSTM 训练
├── ensemble.py             # 统计模型 + 深度模型的集成策略
├── strategy_backtest.py    # A 股交易策略回测
├── diagnostics.py          # 残差诊断（ACF/PACF、Ljung-Box、ARCH）
├── report_builder.py       # 自动生成图表与报告
├── run_full_pipeline.py    # 一键全流程入口
└── outputs/                # 全部产出（tables / figures / reports）
```

## 数据来源

- `data/stock_300502.csv`：个股 OHLCV 日频数据
- `data/benchmark_300.csv`：沪深 300 指数，作为大盘控制变量

数据为本地导出的 CSV，**主流程不依赖联网下载**，20+ 维特征均由本地已有数据构造，保证离线可完整复现。

## 快速开始

```bash
pip install -r requirements.txt

# 一键跑通 L1–L6 全流程
python run_full_pipeline.py
```

完整输出位于：

- `outputs/tables` — 各阶段数值结果（CSV）
- `outputs/figures` — ACF/PACF、脉冲响应、回测净值曲线等
- `outputs/reports` — 汇总 Markdown 与 Word 报告

## 模型权重

| 文件 | 说明 |
|---|---|
| `checkpoints/legacy_best_model.pth` | 早期 10 维特征版本的基线权重 |
| `checkpoints/exam_lstm_best.pth` | 20+ 维特征版本重新训练的权重 |
