import numpy as np

from src.services.simulator import (
    SimulationParams,
    compute_log_returns,
    compute_simple_returns,
    simulate_gbm,
)


class TestSimulateGBM:
    def test_output_shape(self):
        params = SimulationParams(
            initial_price=100, drift=0.05, volatility=0.2, time_steps=252, num_paths=10
        )
        prices = simulate_gbm(params, seed=0)
        assert prices.shape == (10, 253)  # num_paths x (time_steps + 1)

    def test_initial_price(self):
        params = SimulationParams(
            initial_price=50.0, drift=0.0, volatility=0.3, time_steps=100, num_paths=5
        )
        prices = simulate_gbm(params, seed=1)
        np.testing.assert_allclose(prices[:, 0], 50.0)

    def test_prices_positive(self):
        params = SimulationParams(
            initial_price=100, drift=-0.2, volatility=0.5, time_steps=500, num_paths=20
        )
        prices = simulate_gbm(params, seed=2)
        assert np.all(prices > 0), "GBM prices must always be positive"

    def test_reproducible_with_seed(self):
        params = SimulationParams(
            initial_price=100, drift=0.1, volatility=0.2, time_steps=100, num_paths=3
        )
        a = simulate_gbm(params, seed=99)
        b = simulate_gbm(params, seed=99)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        params = SimulationParams(
            initial_price=100, drift=0.1, volatility=0.2, time_steps=100, num_paths=1
        )
        a = simulate_gbm(params, seed=1)
        b = simulate_gbm(params, seed=2)
        assert not np.array_equal(a, b)


class TestReturns:
    def test_log_returns_shape(self, sample_prices):
        lr = compute_log_returns(sample_prices)
        assert lr.shape == (252,)

    def test_simple_returns_shape(self, sample_prices):
        sr = compute_simple_returns(sample_prices)
        assert sr.shape == (252,)

    def test_log_returns_finite(self, sample_prices):
        lr = compute_log_returns(sample_prices)
        assert np.all(np.isfinite(lr))
