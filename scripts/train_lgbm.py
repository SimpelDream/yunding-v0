"""LightGBM训练脚本。

此脚本用于训练LightGBM模型，用于预测胜率和伤害。
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tftassist.predictor.lgbm import LGBMPredictor

logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="训练LightGBM模型")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="数据文件路径"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="保存目录"
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=96,
        help="叶子节点数"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=1200,
        help="树的数量"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.03,
        help="学习率"
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="样本采样比例"
    )
    parser.add_argument(
        "--colsample",
        type=float,
        default=0.8,
        help="特征采样比例"
    )
    return parser.parse_args()

def load_data(data_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载数据。
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        特征矩阵、胜率标签、伤害标签
    """
    # 加载数据
    df = pd.read_csv(data_path)
    
    # 分离特征和标签
    X = df.drop(["win", "damage"], axis=1).values
    y_win = df["win"].values
    y_dmg = df["damage"].values
    
    return X, y_win, y_dmg

def main():
    """主函数。"""
    args = parse_args()
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    X, y_win, y_dmg = load_data(args.data)
    
    # 训练参数
    params = {
        "num_leaves": args.num_leaves,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample
    }
    
    # 训练模型
    LGBMPredictor.train(
        X=X,
        y_win=y_win,
        y_dmg=y_dmg,
        save_dir=save_dir,
        **params
    )
    
    logger.info("训练完成")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main() 