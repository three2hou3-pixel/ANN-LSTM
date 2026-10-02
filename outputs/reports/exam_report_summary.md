# AI 增强型量化选股与回测系统实验摘要

## 数据说明
- 标的: 300502，数据来源: Tushare 已下载本地 CSV。
- 复权方式: 前复权。
- 本项目不依赖新增 Tushare 下载，主要使用本地 OHLCV 与沪深 300 数据构造特征。

## 模型说明
- ARIMA/VAR 作为统计模型骨架，生成预测与残差特征。
- ANN-LSTM 使用 20+ 维新特征重新训练，旧 10 维权重仅作为 legacy baseline 保存。

## 输出文件
- 表格输出位于 `outputs/tables`。
- 图表输出位于 `outputs/figures`。
- 新权重保存为 `checkpoints/exam_lstm_best.pth`。
