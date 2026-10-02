import numpy as np
import pandas as pd
from config import Config


# ==========================================
# 常用技术指标
# ==========================================
def calc_rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / (loss + 1e-8)
    return 100 - 100 / (1 + rs)


def calc_atr(df, window=14):
    high_low = df["High"] - df["Low"]
    high_close_prev = (df["High"] - df["Close"].shift(1)).abs()
    low_close_prev = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def rolling_zscore(series, window=20):
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-8)


# ==========================================
# 30+ 维特征矩阵构建
# ==========================================
def build_feature_table(df_stock, df_bench, arima_resid=None, var_resid=None, config=Config):
    df = df_stock.copy()
    df_b = df_bench.copy().rename(columns={"Close": "Bench_Close"})[["Date", "Bench_Close"]]
    df = pd.merge(df, df_b, on="Date", how="left")
    df["Bench_Close"] = df["Bench_Close"].ffill().bfill()

    df["ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["hs300_ret"] = np.log(df["Bench_Close"] / df["Bench_Close"].shift(1))
    for h in config.target_horizons:
        df[f"target_h{h}"] = np.log(df["Close"].shift(-h) / df["Close"])

    feature_cols = []
    for lag in range(1, 11):
        col = f"ret_lag_{lag}"
        df[col] = df["ret"].shift(lag)
        feature_cols.append(col)

    for win in [5, 10, 20, 60]:
        col = f"MA{win}_gap"
        df[col] = df["Close"] / df["Close"].rolling(win).mean() - 1
        feature_cols.append(col)

    exp12 = df["Close"].ewm(span=12, adjust=False).mean()
    exp26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_DIF"] = exp12 - exp26
    df["MACD_DEA"] = df["MACD_DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD_DIF"] - df["MACD_DEA"]
    df["RSI_6"] = calc_rsi(df["Close"], 6)
    df["RSI_14"] = calc_rsi(df["Close"], 14)
    df["ATR_14"] = calc_atr(df, 14)
    direction = np.sign(df["Close"].diff()).fillna(0)
    df["OBV"] = (direction * df["Volume"]).cumsum()
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_width"] = (4 * bb_std) / (bb_mid + 1e-8)
    feature_cols += ["MACD_DIF", "MACD_DEA", "MACD_HIST", "RSI_6", "RSI_14", "ATR_14", "OBV", "BB_width"]

    for win in [5, 10, 20]:
        col = f"volatility_{win}"
        df[col] = df["ret"].rolling(win).std()
        feature_cols.append(col)

    df["intraday_return"] = np.log(df["Close"] / df["Open"])
    df["high_low_range"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-8)
    df["volume_change"] = np.log(df["Volume"] / df["Volume"].shift(1))
    df["amount_change"] = np.log(df["Amount"] / df["Amount"].shift(1))
    df["turnover"] = df["Turnover"]
    df["turnover_z20"] = rolling_zscore(df["Turnover"], 20)
    df["volume_z20"] = rolling_zscore(df["Volume"], 20)
    feature_cols += ["intraday_return", "high_low_range", "volume_change", "amount_change", "turnover", "turnover_z20", "volume_z20"]

    df["hs300_volatility_20"] = df["hs300_ret"].rolling(20).std()
    df["excess_ret"] = df["ret"] - df["hs300_ret"]
    cov = df["ret"].rolling(20).cov(df["hs300_ret"])
    var = df["hs300_ret"].rolling(20).var()
    df["rolling_beta_20"] = cov / (var + 1e-8)
    df["rolling_corr_20"] = df["ret"].rolling(20).corr(df["hs300_ret"])
    feature_cols += ["hs300_ret", "hs300_volatility_20", "excess_ret", "rolling_beta_20", "rolling_corr_20"]

    df["arima_resid"] = 0.0 if arima_resid is None else pd.Series(arima_resid).reindex(df.index).fillna(0.0)
    df["var_resid"] = 0.0 if var_resid is None else pd.Series(var_resid).reindex(df.index).fillna(0.0)
    for base in ["arima_resid", "var_resid"]:
        for lag in [1, 2]:
            col = f"{base}_lag_{lag}"
            df[col] = df[base].shift(lag)
            feature_cols.append(col)

    scaled_cols = []
    for col in feature_cols:
        scaled = f"{col}_scaled"
        df[scaled] = rolling_zscore(df[col], config.rolling_z_window)
        scaled_cols.append(scaled)

    target_cols = [f"target_h{h}" for h in config.target_horizons]
    df_clean = df.dropna(subset=scaled_cols + target_cols).reset_index(drop=True)
    print(f"特征工程完成：原始特征 {len(feature_cols)} 维，标准化后 {len(scaled_cols)} 维，有效样本 {len(df_clean)} 行。")
    return df_clean, scaled_cols, target_cols


# ==========================================
# LSTM 滑动窗口数据集
# ==========================================
def build_lstm_arrays(df_feature, feature_cols, target_col="target_h1", lookback_window=30):
    X_list, Y_list, date_list = [], [], []
    feature_matrix = df_feature[feature_cols].values
    target_vector = df_feature[target_col].values
    date_vector = df_feature["Date"].values

    for i in range(len(df_feature) - lookback_window + 1):
        X_list.append(feature_matrix[i:i + lookback_window])
        Y_list.append(target_vector[i + lookback_window - 1])
        date_list.append(date_vector[i + lookback_window - 1])

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32).reshape(-1, 1)
    dates = pd.to_datetime(np.array(date_list))
    return X, Y, dates


if __name__ == "__main__":
    from data_loader import load_local_data, clean_market_data
    df_stock, df_bench = load_local_data()
    df_stock, df_bench = clean_market_data(df_stock, df_bench)
    df_feature, feature_cols, target_cols = build_feature_table(df_stock, df_bench)
    X, Y, dates = build_lstm_arrays(df_feature, feature_cols)
    print("X 形状:", X.shape, "Y 形状:", Y.shape, "特征数:", len(feature_cols))
