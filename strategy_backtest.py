import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import Config, ensure_dirs


# ==========================================
# 回测绩效指标
# ==========================================
def calc_performance(equity, position, config=Config):
    equity = np.asarray(equity, dtype=float)
    daily_returns = pd.Series(equity).pct_change().fillna(0).values
    total_return = equity[-1] / equity[0] - 1 if len(equity) > 1 else 0.0
    annual_return = (equity[-1] / equity[0]) ** (config.annual_days / max(len(equity), 1)) - 1 if len(equity) > 1 else 0.0
    annual_vol = np.std(daily_returns) * np.sqrt(config.annual_days)
    sharpe = (np.mean(daily_returns) * config.annual_days - 0.02) / (annual_vol + 1e-8)
    peaks = np.maximum.accumulate(equity)
    mdd = float(np.max((peaks - equity) / (peaks + 1e-8)))
    calmar = annual_return / (mdd + 1e-8)
    win_rate = float(np.mean(daily_returns > 0))
    profit = daily_returns[daily_returns > 0].mean() if np.any(daily_returns > 0) else 0.0
    loss = abs(daily_returns[daily_returns < 0].mean()) if np.any(daily_returns < 0) else 0.0
    turnover = float(np.mean(np.abs(np.diff(position)))) if len(position) > 1 else 0.0
    return {"total_return": total_return, "annual_return": annual_return, "annual_vol": annual_vol, "sharpe": sharpe, "mdd": mdd, "calmar": calmar, "win_rate": win_rate, "profit_loss_ratio": profit / (loss + 1e-8), "turnover": turnover}


# ==========================================
# A 股交易规则回测
# ==========================================
def run_strategy_backtest(pred_df, threshold=0.0, use_vol_weight=False, config=Config):
    df = pred_df.copy().sort_values("Date").reset_index(drop=True)
    pred_col = "ensemble_pred" if "ensemble_pred" in df.columns else "lstm_pred"
    cash = config.initial_cash
    position = 0.0
    equity, position_list = [], []

    for i in range(len(df)):
        signal = 1.0 if df.loc[i, pred_col] > threshold else 0.0
        if use_vol_weight:
            recent_vol = df["true"].iloc[max(0, i - 20):i + 1].std()
            signal *= min(1.0, 0.02 / (recent_vol + 1e-8)) if recent_vol == recent_vol else 1.0
        change = signal - position
        cost = abs(change) * cash * (config.commission_rate + config.slippage)
        if change < 0:
            cost += abs(change) * cash * config.stamp_duty
        cash -= cost
        cash *= 1 + signal * (np.exp(df.loc[i, "true"]) - 1)
        position = signal
        equity.append(cash)
        position_list.append(position)

    result = pd.DataFrame({"Date": df["Date"], "equity": equity, "position": position_list, "true": df["true"]})
    return result, calc_performance(result["equity"].values, result["position"].values, config)


# ==========================================
# 牛熊震荡市态分析
# ==========================================
def market_state_analysis(result, config=Config):
    df = result.copy()
    df["strategy_ret"] = df["equity"].pct_change().fillna(0.0)
    df["market_60"] = df["true"].rolling(60).sum()
    df["market_state"] = np.where(df["market_60"] > 0.10, "bull", np.where(df["market_60"] < -0.10, "bear", "range"))

    rows = []
    for state, part in df.groupby("market_state"):
        if len(part) < 5:
            continue
        temp_equity = (1 + part["strategy_ret"]).cumprod().values
        temp_position = part["position"].values
        metrics = calc_performance(temp_equity, temp_position, config)
        rows.append({"market_state": state, "days": len(part), **metrics})
    state_df = pd.DataFrame(rows)
    state_df.to_csv(config.table_dir / "L6_market_state_performance.csv", index=False, encoding="utf-8-sig")
    return state_df


def run_backtest_suite(pred_df, config=Config):
    ensure_dirs(config)
    result, metrics = run_strategy_backtest(pred_df, 0.0, False, config)
    result.to_csv(config.table_dir / "L6_backtest_equity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(config.table_dir / "L6_backtest_metrics.csv", index=False, encoding="utf-8-sig")

    robust_rows = []
    for threshold in [-0.0005, 0.0, 0.0005]:
        for vol_flag in [False, True]:
            _, m = run_strategy_backtest(pred_df, threshold, vol_flag, config)
            robust_rows.append({"threshold": threshold, "use_vol_weight": vol_flag, "sharpe": m["sharpe"], "annual_return": m["annual_return"], "mdd": m["mdd"]})
    pd.DataFrame(robust_rows).to_csv(config.table_dir / "L6_robustness.csv", index=False, encoding="utf-8-sig")
    market_state_analysis(result, config)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result["Date"], result["equity"] / config.initial_cash, label="AI 策略净值", color="red")
    ax.set_title("样本外回测净值曲线")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.figure_dir / "L6_backtest_equity.png", dpi=300)
    plt.close(fig)
    print(f"回测完成：Sharpe={metrics['sharpe']:.4f}, MDD={metrics['mdd']:.2%}")
    return result, metrics
