import pandas as pd
from config import Config, ensure_dirs
from statistical_models import diebold_mariano, regression_metrics


# ==========================================
# L5 集成预测与 DM 检验
# ==========================================
def build_ensemble(arima_pred, var_pred, lstm_pred, config=Config):
    ensure_dirs(config)
    df = lstm_pred.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    if arima_pred is not None and not arima_pred.empty:
        df = pd.merge(df, arima_pred[["Date", "arima_pred"]], on="Date", how="left")
    else:
        df["arima_pred"] = 0.0
    if var_pred is not None and not var_pred.empty:
        df = pd.merge(df, var_pred[["Date", "var_pred"]], on="Date", how="left")
    else:
        df["var_pred"] = 0.0

    pred_cols = ["lstm_pred", "arima_pred", "var_pred"]
    df[pred_cols] = df[pred_cols].fillna(0.0)
    df["ensemble_pred"] = df[pred_cols].mean(axis=1)

    rows = []
    for col in pred_cols + ["ensemble_pred"]:
        row = {"model": col}
        row.update(regression_metrics(df["true"], df[col]))
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(config.table_dir / "L5_model_compare_metrics.csv", index=False, encoding="utf-8-sig")

    dm_rows = []
    for col in ["arima_pred", "var_pred", "ensemble_pred"]:
        dm_rows.append({"compare": f"lstm_pred vs {col}", **diebold_mariano(df["true"], df["lstm_pred"], df[col])})
    pd.DataFrame(dm_rows).to_csv(config.table_dir / "L5_dm_tests.csv", index=False, encoding="utf-8-sig")
    df.to_csv(config.table_dir / "L5_ensemble_pred.csv", index=False, encoding="utf-8-sig")
    print("集成预测完成：已输出模型对比指标与 DM 检验。")
    return df, metrics
