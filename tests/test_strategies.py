import numpy as np

from src.services.strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    RandomStrategy,
    Signal,
)


class TestMeanReversionStrategy:
    def test_signal_count_matches_prices(self, sample_prices):
        strategy = MeanReversionStrategy(window=20)
        signals = strategy.generate_signals(sample_prices)
        assert len(signals) == len(sample_prices)

    def test_hold_during_warmup(self, sample_prices):
        strategy = MeanReversionStrategy(window=20)
        signals = strategy.generate_signals(sample_prices)
        assert all(s == Signal.HOLD for s in signals[:20])

    def test_generates_trades(self, sample_prices):
        strategy = MeanReversionStrategy(window=20, buy_threshold=-0.01, sell_threshold=0.01)
        signals = strategy.generate_signals(sample_prices)
        non_hold = [s for s in signals if s != Signal.HOLD]
        assert len(non_hold) > 0, "Should generate at least some buy/sell signals"

    def test_name_and_params(self):
        s = MeanReversionStrategy(window=30, buy_threshold=-0.05, sell_threshold=0.05)
        assert s.name == "mean_reversion"
        assert s.params["window"] == 30


class TestMomentumStrategy:
    def test_signal_count_matches_prices(self, sample_prices):
        strategy = MomentumStrategy(lookback=10)
        signals = strategy.generate_signals(sample_prices)
        assert len(signals) == len(sample_prices)

    def test_hold_during_warmup(self, sample_prices):
        strategy = MomentumStrategy(lookback=10)
        signals = strategy.generate_signals(sample_prices)
        assert all(s == Signal.HOLD for s in signals[:10])

    def test_name_and_params(self):
        s = MomentumStrategy(lookback=15, threshold=0.05)
        assert s.name == "momentum"
        assert s.params["lookback"] == 15


class TestRandomStrategy:
    def test_signal_count_matches_prices(self, sample_prices):
        strategy = RandomStrategy(seed=42)
        signals = strategy.generate_signals(sample_prices)
        assert len(signals) == len(sample_prices)

    def test_reproducible(self, sample_prices):
        a = RandomStrategy(seed=42).generate_signals(sample_prices)
        b = RandomStrategy(seed=42).generate_signals(sample_prices)
        assert a == b

    def test_contains_buys_and_sells(self):
        prices = np.ones(1000)  # flat prices, strategy is random anyway
        strategy = RandomStrategy(buy_probability=0.1, sell_probability=0.1, seed=42)
        signals = strategy.generate_signals(prices)
        assert Signal.BUY in signals
        assert Signal.SELL in signals
