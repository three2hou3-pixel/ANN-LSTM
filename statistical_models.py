import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from config import Config, ensure_dirs

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
except Exception:
    ARIMA = acorr_ljungbox = het_arch = VAR = coint_johansen = None


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return {
        "direction_acc": float(np.mean(np.sign(y_pred) == np.sign(y_true))),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
    }


# ==========================================
# L3 ARIMA 网格搜索与滚动预测
# ==========================================
def run_arima_analysis(df_feature, config=Config, max_p=5, max_q=5):
    ensure_dirs(config)
    if ARIMA is None:
        print("statsmodels 不可用，跳过 ARIMA。")
        return {"candidates": pd.DataFrame(), "rolling_pred": pd.DataFrame(), "resid": pd.Series(0.0, index=df_feature.index)}

    r = df_feature.set_index("Date")["ret"].dropna()
    train_r = r[r.index <= pd.Timestamp(config.train_end)]
    test_r = r[(r.index >= pd.Timestamp(config.test_start)) & (r.index <= pd.Timestamp(config.test_end))]

    rows = []
    warnings.filterwarnings("ignore")
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                fit = ARIMA(train_r, order=(p, 0, q)).fit()
                rows.append({"p": p, "d": 0, "q": q, "aic": fit.aic, "bic": fit.bic})
            except Exception:
                pass
    candidates = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    candidates.head(3).to_csv(config.table_dir / "L3_arima_top3.csv", index=False, encoding="utf-8-sig")

    if candidates.empty or len(test_r) == 0:
        return {"candidates": candidates, "rolling_pred": pd.DataFrame(), "resid": pd.Series(0.0, index=df_feature.index)}

    best = candidates.iloc[0]
    order = (int(best.p), 0, int(best.q))
    fitted = ARIMA(train_r, order=order).fit()
    resid_in_sample = fitted.resid

    history = list(train_r.values)
    preds = []
    for value in test_r.values:
        try:
            preds.append(float(ARIMA(history, order=order).fit().forecast()[0]))
        except Exception:
            preds.append(0.0)
        history.append(value)

    rolling_pred = pd.DataFrame({"Date": test_r.index, "true": test_r.values, "arima_pred": preds})
    metrics = regression_metrics(rolling_pred["true"], rolling_pred["arima_pred"])
    rolling_pred.to_csv(config.table_dir / "L3_arima_rolling_pred.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(config.table_dir / "L3_arima_metrics.csv", index=False, encoding="utf-8-sig")

    if acorr_ljungbox is not None:
        acorr_ljungbox(resid_in_sample.dropna(), lags=[10, 20], return_df=True).to_csv(config.table_dir / "L3_ljung_box.csv", encoding="utf-8-sig")
    if het_arch is not None:
        arch = het_arch(resid_in_sample.dropna(), nlags=10)
        pd.DataFrame([{"test": "ARCH-LM", "stat": arch[0], "pvalue": arch[1]}]).to_csv(config.table_dir / "L3_arch_effect.csv", index=False, encoding="utf-8-sig")

    resid = pd.Series(0.0, index=df_feature.index)
    pos = {d: i for i, d in enumerate(df_feature["Date"])}
    for d, v in resid_in_sample.items():
        if d in pos:
            resid.iloc[pos[d]] = v

    print(f"ARIMA 完成：最优阶数 {order}，测试方向准确率 {metrics['direction_acc']:.2%}")
    return {"candidates": candidates, "rolling_pred": rolling_pred, "resid": resid}


# ==========================================
# L4 VAR / 协整 / Granger / IRF
# ==========================================
def run_var_analysis(df_feature, config=Config):
    ensure_dirs(config)
    if VAR is None:
        print("statsmodels 不可用，跳过 VAR。")
        return {"var_pred": pd.DataFrame(), "resid": pd.Series(0.0, index=df_feature.index)}

    var_df = df_feature[["Date", "ret", "hs300_ret", "volume_change", "volatility_20"]].dropna().set_index("Date")
    train_df = var_df[var_df.index <= pd.Timestamp(config.train_end)]
    test_df = var_df[(var_df.index >= pd.Timestamp(config.test_start)) & (var_df.index <= pd.Timestamp(config.test_end))]

    if coint_johansen is not None:
        try:
            joh = coint_johansen(train_df[["ret", "hs300_ret"]], det_order=0, k_ar_diff=1)
            pd.DataFrame([{"trace_stat_0": joh.lr1[0], "crit_95_0": joh.cvt[0, 1]}]).to_csv(config.table_dir / "L4_johansen_test.csv", index=False, encoding="utf-8-sig")
        except Exception:
            pass

    model = VAR(train_df)
    selected = model.select_order(maxlags=5)
    lag = int(selected.aic) if selected.aic and selected.aic > 0 else 1
    lag = max(1, min(lag, 5))
    fitted = model.fit(lag)

    granger_rows = []
    for cause in ["hs300_ret", "volume_change", "volatility_20"]:
        try:
            res = fitted.test_causality("ret", [cause], kind="f")
            granger_rows.append({"cause": cause, "stat": res.test_statistic, "pvalue": res.pvalue})
        except Exception:
            pass
    pd.DataFrame(granger_rows).to_csv(config.table_dir / "L4_granger_tests.csv", index=False, encoding="utf-8-sig")

    try:
        fig = fitted.irf(10).plot(orth=False)
        fig.savefig(config.figure_dir / "L4_var_irf.png", dpi=300)
        plt.close(fig)
    except Exception:
        pass

    history = train_df.values.tolist()
    pred_rows = []
    for date, row in test_df.iterrows():
        try:
            pred = fitted.forecast(np.array(history[-lag:]), steps=1)[0][0]
        except Exception:
            pred = 0.0
        pred_rows.append({"Date": date, "true": row["ret"], "var_pred": pred})
        history.append(row.values.tolist())
    var_pred = pd.DataFrame(pred_rows)
    var_pred.to_csv(config.table_dir / "L4_var_rolling_pred.csv", index=False, encoding="utf-8-sig")

    resid_series = pd.Series(0.0, index=df_feature.index)
    pos = {d: i for i, d in enumerate(df_feature["Date"])}
    for d, v in fitted.resid["ret"].items():
        if d in pos:
            resid_series.iloc[pos[d]] = v

    print(f"VAR 完成：滞后阶数 {lag}，输出 Granger 与 IRF。")
    return {"var_pred": var_pred, "resid": resid_series}


def diebold_mariano(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true).flatten()
    pred_a = np.asarray(pred_a).flatten()
    pred_b = np.asarray(pred_b).flatten()
    d = (y_true - pred_a) ** 2 - (y_true - pred_b) ** 2
    dm_stat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-8)
    pvalue = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return {"dm_stat": float(dm_stat), "pvalue": float(pvalue)}
