from pathlib import Path
import torch


# ==========================================
# 考试版项目统一配置
# ==========================================
class Config:
    # 标的与数据
    stock_code = "300502"
    stock_name = "新易盛"
    data_source = "Tushare 已下载本地 CSV"
    adjustment = "前复权"

    # 路径
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    checkpoint_dir = base_dir / "checkpoints"
    output_dir = base_dir / "outputs"
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    report_dir = output_dir / "reports"

    # 时间切分：测试集超过 12 个月，且与训练期不重叠
    train_end = "2023-12-31"
    val_start = "2024-01-01"
    val_end = "2024-12-31"
    test_start = "2025-01-01"
    test_end = "2026-05-31"

    # 特征与标签
    lookback_window = 30
    rolling_z_window = 60
    target_horizons = [1, 5, 20]
    main_horizon = 1

    # LSTM 训练
    batch_size = 64
    learning_rate = 1e-3
    weight_decay = 1e-4
    max_epochs = 80
    early_stop_patience = 12
    n_features = None
    lstm_hidden_dim = 64
    ann_hidden_dim = 64
    dropout = 0.2
    alpha = 2.0
    delta = 0.05
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A 股交易成本
    initial_cash = 100000.0
    commission_rate = 0.00025
    stamp_duty = 0.0005
    slippage = 0.001
    annual_days = 242


def ensure_dirs(config=Config):
    for path in [config.checkpoint_dir, config.figure_dir, config.table_dir, config.report_dir]:
        path.mkdir(parents=True, exist_ok=True)
