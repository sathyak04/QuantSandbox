import numpy as np
import pytest

from src.services.simulator import SimulationParams, simulate_gbm


@pytest.fixture
def sample_prices() -> np.ndarray:
    """A single price path with 252 steps for testing."""
    params = SimulationParams(
        initial_price=100.0, drift=0.05, volatility=0.2, time_steps=252, num_paths=1
    )
    return simulate_gbm(params, seed=42)[0]


@pytest.fixture
def long_prices() -> np.ndarray:
    """A longer price path (500 steps) needed for ML strategy."""
    params = SimulationParams(
        initial_price=100.0, drift=0.05, volatility=0.2, time_steps=500, num_paths=1
    )
    return simulate_gbm(params, seed=42)[0]
