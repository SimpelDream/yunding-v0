"""LightGBM 训练脚本。

此脚本用于训练胜率和伤害预测模型。
"""

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def main() -> None:
    """主函数。"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 加载数据
    data_path = Path("datasets/train.csv")
    df = pd.read_csv(data_path)

    # 准备特征和标签
    X = df.drop(["win_rate", "damage"], axis=1)
    y = df[["win_rate", "damage"]]

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 设置参数
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 96,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    # 训练模型
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1200,
        valid_sets=[train_data, val_data],
        early_stopping_rounds=50,
    )

    # 保存模型
    model.save_model("models/lgbm.txt")

    # 评估模型
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
    val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))

    logger.info(f"训练集 RMSE: {train_rmse:.4f}")
    logger.info(f"验证集 RMSE: {val_rmse:.4f}")


if __name__ == "__main__":
    main() 