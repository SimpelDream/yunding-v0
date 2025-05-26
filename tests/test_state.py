"""游戏状态测试模块。"""

import pytest
from typing import List

from tftassist.core.state import BoardState, Unit

def test_board_state_init() -> None:
    """测试BoardState初始化。"""
    state = BoardState(
        side="self",
        stage="1-1",
        rank=1,
        hp=100,
        gold=0,
        level=1,
        xp_progress=(0, 2),
        timestamp=0.0
    )
    
    assert state.side == "self"
    assert state.stage == "1-1"
    assert state.rank == 1
    assert state.hp == 100
    assert state.gold == 0
    assert state.level == 1
    assert state.xp_progress == (0, 2)
    assert state.timestamp == 0.0
    assert state.units == []
    assert state.bench_units == []
    assert state.traits == {}
    assert state.inactive_traits == {}
    assert state.augments == []
    assert state.hex_map == {}
    assert state.combo_meter == 0
    assert state.enemy_hp_vec == []
    assert state.fps == 0
    assert state.ping_ms == 0

def test_unit_init() -> None:
    """测试Unit初始化。"""
    unit = Unit(
        name="Ahri",
        star=2,
        items=["Rabadon's Deathcap"],
        position=(0, 0),
        hp_pct=1.0,
        shield_pct=0.0,
        status_tags=["Charmed"]
    )
    
    assert unit.name == "Ahri"
    assert unit.star == 2
    assert unit.items == ["Rabadon's Deathcap"]
    assert unit.position == (0, 0)
    assert unit.hp_pct == 1.0
    assert unit.shield_pct == 0.0
    assert unit.status_tags == ["Charmed"]

def test_board_state_update() -> None:
    """测试BoardState更新。"""
    state = BoardState(
        side="self",
        stage="1-1",
        rank=1,
        hp=100,
        gold=0,
        level=1,
        xp_progress=(0, 2),
        timestamp=0.0
    )
    
    # 添加单位
    unit = Unit(
        name="Ahri",
        star=2,
        items=["Rabadon's Deathcap"],
        position=(0, 0),
        hp_pct=1.0,
        shield_pct=0.0,
        status_tags=["Charmed"]
    )
    state.units.append(unit)
    
    # 添加特质
    state.traits["Mage"] = 3
    state.inactive_traits["Assassin"] = 2
    
    # 添加强化符文
    state.augments.append("Mage Heart")
    
    # 添加六边形
    state.hex_map[(0, 0)] = "Mage"
    
    # 更新其他属性
    state.combo_meter = 3
    state.enemy_hp_vec = [100, 80, 60]
    state.fps = 60
    state.ping_ms = 20
    
    assert len(state.units) == 1
    assert state.units[0].name == "Ahri"
    assert state.traits["Mage"] == 3
    assert state.inactive_traits["Assassin"] == 2
    assert state.augments == ["Mage Heart"]
    assert state.hex_map[(0, 0)] == "Mage"
    assert state.combo_meter == 3
    assert state.enemy_hp_vec == [100, 80, 60]
    assert state.fps == 60
    assert state.ping_ms == 20

def test_unit_creation():
    """测试单位创建。"""
    # 有效单位
    unit = Unit(
        name="test_unit",
        star=2,
        items=["item1", "item2"],
        position=(3, 2),
        hp_pct=0.8,
        shield_pct=0.3,
        status_tags=["stunned"]
    )
    assert unit.name == "test_unit"
    assert unit.star == 2
    assert len(unit.items) == 2
    assert unit.position == (3, 2)
    assert unit.hp_pct == 0.8
    assert unit.shield_pct == 0.3
    assert len(unit.status_tags) == 1
    
    # 无效星级
    with pytest.raises(ValidationError):
        Unit(
            name="test_unit",
            star=4,  # 无效星级
            items=[],
            position=(0, 0),
            hp_pct=1.0,
            shield_pct=0.0,
            status_tags=[]
        )
    
    # 无效生命值
    with pytest.raises(ValidationError):
        Unit(
            name="test_unit",
            star=1,
            items=[],
            position=(0, 0),
            hp_pct=1.5,  # 无效生命值
            shield_pct=0.0,
            status_tags=[]
        )

def test_board_state_creation():
    """测试棋盘状态创建。"""
    # 有效状态
    state = BoardState(
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
        units=[],
        bench_units=[],
        augments=["augment1"],
        hex_map={(0, 0): "blue_hex"},
        combo_meter=100,
        enemy_hp_vec=[100, 90, 80],
        fps=60.0,
        ping_ms=20.0,
        timestamp=1234567890.0
    )
    assert state.side == "self"
    assert state.stage == "4-2"
    assert state.phase_timer == 30
    assert state.rank == 1
    assert state.hp == 100
    assert state.gold == 50
    assert state.level == 8
    assert state.xp_progress == (80, 100)
    assert len(state.shop_odds) == 3
    assert len(state.shop_cards) == 2
    assert len(state.traits) == 2
    assert len(state.inactive_traits) == 1
    assert len(state.units) == 0
    assert len(state.bench_units) == 0
    assert len(state.augments) == 1
    assert len(state.hex_map) == 1
    assert state.combo_meter == 100
    assert len(state.enemy_hp_vec) == 3
    assert state.fps == 60.0
    assert state.ping_ms == 20.0
    assert state.timestamp == 1234567890.0
    
    # 无效阵营
    with pytest.raises(ValidationError):
        BoardState(
            side="invalid",  # 无效阵营
            stage="4-2",
            phase_timer=30,
            rank=1,
            hp=100,
            gold=50,
            level=8,
            xp_progress=(80, 100),
            shop_odds={},
            shop_cards=[],
            traits={},
            inactive_traits={},
            units=[],
            bench_units=[],
            augments=[],
            hex_map={},
            combo_meter=None,
            enemy_hp_vec=[],
            fps=None,
            ping_ms=None,
            timestamp=1234567890.0
        ) 