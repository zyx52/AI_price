"""
pytest 共享 fixtures 和配置
"""
import pytest
import numpy as np
import pandas as pd

from data import DataLoader


@pytest.fixture(scope="session")
def mock_history() -> pd.DataFrame:
    """共享的历史数据 fixture, 整个测试会话只加载一次"""
    loader = DataLoader(source="mock")
    return loader.load_history("2024-01-01", "2025-12-31")


@pytest.fixture(scope="session")
def mock_external_signal() -> dict:
    """共享的外部信号 fixture"""
    loader = DataLoader(source="mock")
    return loader.load_external_signal("2026-05-02")


@pytest.fixture
def rng() -> np.random.Generator:
    """可重现的随机数生成器"""
    return np.random.default_rng(42)
