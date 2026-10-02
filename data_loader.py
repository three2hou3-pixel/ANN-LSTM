import pandas as pd
from config import Config


# ==========================================
# 本地数据读取
# ==========================================
def load_local_data(config=Config):
    """读取已经下载好的 Tushare CSV，不再依赖额外接口权限。"""
    stock_path = config.data_dir / f"stock_{config.stock_code}.csv"
    bench_path = config.data_dir / "benchmark_300.csv"

    df_stock = pd.read_csv(stock_path, parse_dates=["Date"])
    df_bench = pd.read_csv(bench_path, parse_dates=["Date"])

    df_stock = df_stock.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    df_bench = df_bench.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

    df_stock = df_stock[df_stock["Date"] <= pd.Timestamp(config.test_end)].reset_index(drop=True)
    df_bench = df_bench[df_bench["Date"] <= pd.Timestamp(config.test_end)].reset_index(drop=True)

    print(f"个股数据: {df_stock['Date'].min().date()} ~ {df_stock['Date'].max().date()}, 共 {len(df_stock)} 行")
    print(f"基准数据: {df_bench['Date'].min().date()} ~ {df_bench['Date'].max().date()}, 共 {len(df_bench)} 行")
    return df_stock, df_bench


# ==========================================
# 缺失值与异常值处理
# ==========================================
def clean_market_data(df_stock, df_bench):
    """
    金融日频数据不适合粗暴删除太多行。
    本项目采用：日期排序 + 前后向填充 + 1%/99% 分位缩尾，保留样本连续性。
    """
    df_stock = df_stock.copy()
    df_bench = df_bench.copy()

    stock_numeric = ["Open", "High", "Low", "Close", "Volume", "Amount", "Turnover"]
    bench_numeric = ["Close"]

    for col in stock_numeric:
        df_stock[col] = pd.to_numeric(df_stock[col], errors="coerce")
    for col in bench_numeric:
        df_bench[col] = pd.to_numeric(df_bench[col], errors="coerce")

    before_na = int(df_stock[stock_numeric].isna().sum().sum() + df_bench[bench_numeric].isna().sum().sum())
    df_stock[stock_numeric] = df_stock[stock_numeric].ffill().bfill()
    df_bench[bench_numeric] = df_bench[bench_numeric].ffill().bfill()

    for col in stock_numeric:
        low, high = df_stock[col].quantile([0.01, 0.99])
        df_stock[col] = df_stock[col].clip(lower=low, upper=high)

    print(f"缺失值处理完成，原始缺失单元格数量: {before_na}")
    print("异常值处理策略: 对个股数值列进行 1%/99% 分位缩尾。")
    print("复权方式说明: 本地个股价格由原 get_data.py 按前复权逻辑生成。")
    return df_stock, df_bench


if __name__ == "__main__":
    df_stock, df_bench = load_local_data()
    df_stock, df_bench = clean_market_data(df_stock, df_bench)
    print(df_stock.head())
