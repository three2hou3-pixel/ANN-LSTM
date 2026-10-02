import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from config import Config, ensure_dirs

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
except Exception:
    adfuller = kpss = plot_acf = plot_pacf = None


# ==========================================
# L2 描述统计与平稳性检验
# ==========================================
def run_diagnostics(df_feature, config=Config):
    ensure_dirs(config)
    r = df_feature["ret"].dropna()
    jb_stat, jb_p = stats.jarque_bera(r)

    desc = pd.DataFrame([{
        "mean": r.mean(),
        "std": r.std(),
        "skew": r.skew(),
        "kurtosis": r.kurtosis(),
        "JB_stat": jb_stat,
        "JB_pvalue": jb_p,
    }])

    rows = []
    if adfuller is not None:
        adf = adfuller(r, autolag="AIC")
        rows.append({"test": "ADF", "stat": adf[0], "pvalue": adf[1]})
    if kpss is not None:
        try:
            kp = kpss(r, regression="c", nlags="auto")
            rows.append({"test": "KPSS", "stat": kp[0], "pvalue": kp[1]})
        except Exception as e:
            rows.append({"test": "KPSS", "stat": np.nan, "pvalue": np.nan, "note": str(e)})
    stationarity = pd.DataFrame(rows)

    tmp = df_feature[["Date", "ret"]].dropna().copy()
    tmp["weekday"] = tmp["Date"].dt.day_name()
    calendar = tmp.groupby("weekday")["ret"].agg(["mean", "std", "count", lambda x: (x > 0).mean()]).reset_index()
    calendar.columns = ["weekday", "mean", "std", "count", "positive_rate"]

    desc.to_csv(config.table_dir / "L2_descriptive_stats.csv", index=False, encoding="utf-8-sig")
    stationarity.to_csv(config.table_dir / "L2_stationarity_tests.csv", index=False, encoding="utf-8-sig")
    calendar.to_csv(config.table_dir / "L2_calendar_effect.csv", index=False, encoding="utf-8-sig")

    if plot_acf is not None and plot_pacf is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(r, ax=axes[0], lags=40)
        plot_pacf(r, ax=axes[1], lags=40, method="ywm")
        axes[0].set_title("收益率 ACF")
        axes[1].set_title("收益率 PACF")
        fig.tight_layout()
        fig.savefig(config.figure_dir / "L2_acf_pacf.png", dpi=300)
        plt.close(fig)

    print("L2 诊断完成：描述统计、平稳性检验、日历效应、ACF/PACF 已输出。")
    return {"desc": desc, "stationarity": stationarity, "calendar": calendar}
