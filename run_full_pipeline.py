from config import Config, ensure_dirs
from data_loader import load_local_data, clean_market_data
from feature_engineering import build_feature_table, build_lstm_arrays
from diagnostics import run_diagnostics
from statistical_models import run_arima_analysis, run_var_analysis
from feature_selection import run_feature_selection
from train_exam_lstm import train_exam_lstm
from ensemble import build_ensemble
from strategy_backtest import run_backtest_suite
from report_builder import build_report_summary


# ==========================================
# 一键运行完整考试版流程
# ==========================================
def main():
    ensure_dirs(Config)

    print("=" * 60)
    print("  L1 数据读取与清洗")
    print("=" * 60)
    df_stock, df_bench = load_local_data(Config)
    df_stock, df_bench = clean_market_data(df_stock, df_bench)

    print("\n" + "=" * 60)
    print("  第一轮特征工程：先生成无残差占位特征")
    print("=" * 60)
    df_feature, feature_cols, target_cols = build_feature_table(df_stock, df_bench, config=Config)

    print("\n" + "=" * 60)
    print("  L2 探索性分析与平稳性检验")
    print("=" * 60)
    run_diagnostics(df_feature, Config)

    print("\n" + "=" * 60)
    print("  L3/L4 统计模型")
    print("=" * 60)
    arima_result = run_arima_analysis(df_feature, Config)
    var_result = run_var_analysis(df_feature, Config)

    print("\n" + "=" * 60)
    print("  第二轮特征工程：并入 ARIMA/VAR 残差特征")
    print("=" * 60)
    df_feature, feature_cols, target_cols = build_feature_table(
        df_stock,
        df_bench,
        arima_resid=arima_result["resid"],
        var_resid=var_result["resid"],
        config=Config
    )
    run_feature_selection(df_feature, feature_cols, target_col=f"target_h{Config.main_horizon}", config=Config)
    X, Y, dates = build_lstm_arrays(
        df_feature,
        feature_cols,
        target_col=f"target_h{Config.main_horizon}",
        lookback_window=Config.lookback_window
    )

    print("\n" + "=" * 60)
    print("  L5 训练考试版 ANN-LSTM")
    print("=" * 60)
    model, history, lstm_pred, test_metrics = train_exam_lstm(X, Y, dates, len(feature_cols), Config)

    print("\n" + "=" * 60)
    print("  L5 集成预测")
    print("=" * 60)
    ensemble_pred, compare_metrics = build_ensemble(arima_result["rolling_pred"], var_result["var_pred"], lstm_pred, Config)

    print("\n" + "=" * 60)
    print("  L6 策略回测")
    print("=" * 60)
    run_backtest_suite(ensemble_pred, Config)

    print("\n" + "=" * 60)
    print("  报告摘要")
    print("=" * 60)
    build_report_summary(Config)
    print("完整流程运行结束。")


if __name__ == "__main__":
    main()
