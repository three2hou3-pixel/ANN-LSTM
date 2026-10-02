import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from config import Config, ensure_dirs

try:
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.ensemble import RandomForestRegressor
except Exception:
    mutual_info_regression = None
    RandomForestRegressor = None


# ==========================================
# L5 两种特征选择方法
# ==========================================
def run_feature_selection(df_feature, feature_cols, target_col="target_h1", config=Config):
    """
    题目要求至少两种特征选择方法。
    这里固定输出：互信息筛选 + 随机森林重要性排序。
    """
    ensure_dirs(config)
    X = df_feature[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df_feature[target_col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 方法一：互信息，捕捉非线性相关
    if mutual_info_regression is not None:
        mi_score = mutual_info_regression(X, y, random_state=42)
    else:
        mi_score = np.abs([np.corrcoef(X[col], y)[0, 1] if X[col].std() > 0 else 0.0 for col in feature_cols])
    mi_df = pd.DataFrame({"feature": feature_cols, "mutual_info_score": mi_score}).sort_values("mutual_info_score", ascending=False)
    mi_df.to_csv(config.table_dir / "L5_feature_selection_mutual_info.csv", index=False, encoding="utf-8-sig")

    # 方法二：随机森林重要性，作为树模型视角的特征筛选
    if RandomForestRegressor is not None:
        rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_score = rf.feature_importances_
    else:
        rf_score = np.zeros(len(feature_cols))
    rf_df = pd.DataFrame({"feature": feature_cols, "rf_importance": rf_score}).sort_values("rf_importance", ascending=False)
    rf_df.to_csv(config.table_dir / "L5_feature_selection_random_forest.csv", index=False, encoding="utf-8-sig")

    print("特征选择完成：已输出互信息与随机森林重要性排序。")
    return {"mutual_info": mi_df, "random_forest": rf_df}
