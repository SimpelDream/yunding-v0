"""预测器测试模块。"""

import os
from pathlib import Path

import numpy as np
import pytest

from tftassist.core.state import BoardState, Unit
from tftassist.predictor.lgbm import LGBMPredictor

@pytest.fixture
def predictor():
    """创建预测器实例。"""
    win_model_path = Path("models/win_model.txt")
    dmg_model_path = Path("models/dmg_model.txt")
    if not win_model_path.exists() or not dmg_model_path.exists():
        pytest.skip("模型文件不存在")
    return LGBMPredictor(win_model_path, dmg_model_path)

@pytest.fixture
def test_state():
    """创建测试状态。"""
    return BoardState(
        side="self",
        stage="4-2",
        phase_timer=30,
        rank=1,
        hp=100,
        gold=50,
        level=8,
        xp_progress=(80, 100),
        shop_odds={"1": 0.1, "2": 0.2, "3": 0.3},
        shop_cards=["card1", "card2"],
        traits={"trait1": 3, "trait2": 2},
        inactive_traits={"trait3": 1},
        units=[
            Unit(
                name="unit1",
                star=2,
                items=["item1"],
                position=(3, 2),
                hp_pct=0.8,
                shield_pct=0.3,
                status_tags=[]
            )
        ],
        bench_units=[
            Unit(
                name="unit2",
                star=1,
                items=[],
                position=(0, 0),
                hp_pct=1.0,
                shield_pct=0.0,
                status_tags=[]
            )
        ],
        augments=["augment1"],
        hex_map={(0, 0): "blue_hex"},
        combo_meter=100,
        enemy_hp_vec=[100, 90, 80],
        fps=60.0,
        ping_ms=20.0,
        timestamp=1234567890.0
    )

def test_predictor_initialization(predictor):
    """测试预测器初始化。"""
    assert predictor.win_model is not None
    assert predictor.dmg_model is not None
    assert isinstance(predictor.feature_names, list)

def test_feature_building(predictor, test_state):
    """测试特征构建。"""
    features = predictor.build_feature(test_state)
    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float32
    assert features.ndim == 1

def test_prediction(predictor, test_state):
    """测试预测。"""
    features = predictor.build_feature(test_state)
    win_prob, dmg = predictor.predict(features)
    assert isinstance(win_prob, float)
    assert isinstance(dmg, float)
    assert 0 <= win_prob <= 1
    assert dmg >= 0

def test_training(tmp_path):
    """测试模型训练。"""
    # 创建测试数据
    X = np.random.rand(100, 50)
    y_win = np.random.rand(100)
    y_dmg = np.random.rand(100)
    
    # 创建保存目录
    save_dir = tmp_path / "models"
    
    # 训练模型
    LGBMPredictor.train(
        X=X,
        y_win=y_win,
        y_dmg=y_dmg,
        save_dir=save_dir,
        num_leaves=31,
        n_estimators=100,
        learning_rate=0.1
    )
    
    # 检查输出
    assert save_dir.exists()
    assert (save_dir / "win_model.txt").exists()
    assert (save_dir / "dmg_model.txt").exists()

def test_unit_feature_extraction(predictor):
    """测试单位特征提取。"""
    units = [
        {
            "name": "unit1",
            "star": 2,
            "items": ["item1", "item2"],
            "position": (3, 2),
            "hp_pct": 0.8,
            "shield_pct": 0.3,
            "status_tags": []
        },
        {
            "name": "unit2",
            "star": 3,
            "items": ["item3"],
            "position": (4, 1),
            "hp_pct": 1.0,
            "shield_pct": 0.0,
            "status_tags": []
        }
    ]
    
    features = predictor._extract_unit_features(units)
    assert isinstance(features, list)
    assert len(features) > 0
    assert all(isinstance(f, float) for f in features)

def test_hex_feature_extraction(predictor):
    """测试海克斯特征提取。"""
    hex_map = {
        (0, 0): "blue_hex",
        (1, 1): "artifact_hex",
        (2, 2): "void_hex"
    }
    
    features = predictor._extract_hex_features(hex_map)
    assert isinstance(features, list)
    assert len(features) > 0
    assert all(isinstance(f, float) for f in features) 