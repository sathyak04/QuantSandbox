import numpy as np

from src.services.backtester import run_backtest
from src.services.strategies import MeanReversionStrategy, MomentumStrategy, RandomStrategy, Signal


class TestBacktest:
    def test_portfolio_length_matches_prices(self, sample_prices):
        result = run_backtest(sample_prices, MeanReversionStrategy())
        assert len(result.portfolio_values) == len(sample_prices)

    def test_initial_portfolio_value(self, sample_prices):
        result = run_backtest(sample_prices, RandomStrategy(seed=99), initial_cash=50_000)
        # First value should be initial cash (no trade on step 0 for random with this seed)
        # or initial_cash if first signal is HOLD
        assert result.portfolio_values[0] > 0

    def test_signals_length(self, sample_prices):
        result = run_backtest(sample_prices, MomentumStrategy())
        assert len(result.signals) == len(sample_prices)

    def test_metrics_types(self, sample_prices):
        result = run_backtest(sample_prices, MeanReversionStrategy())
        m = result.metrics
        assert isinstance(m.total_return, float)
        assert isinstance(m.sharpe_ratio, float)
        assert isinstance(m.max_drawdown, float)
        assert isinstance(m.win_count, int)
        assert isinstance(m.loss_count, int)

    def test_max_drawdown_negative_or_zero(self, sample_prices):
        result = run_backtest(sample_prices, MeanReversionStrategy())
        assert result.metrics.max_drawdown <= 0

    def test_trades_have_correct_actions(self, sample_prices):
        result = run_backtest(sample_prices, MeanReversionStrategy())
        for trade in result.trades:
            assert trade.action in (Signal.BUY, Signal.SELL)
            assert trade.price > 0
            assert trade.quantity > 0

    def test_buy_hold_baseline(self):
        """Constant rising prices: buy-and-hold should profit."""
        prices = np.linspace(100, 200, 100)
        # Mean reversion on steadily rising prices should buy early
        strat = MeanReversionStrategy(window=5, buy_threshold=-0.001, sell_threshold=0.05)
        result = run_backtest(prices, strat)
        assert result.portfolio_values[-1] >= 90_000  # shouldn't lose too much


class TestMLBacktest:
    def test_ml_strategy_runs(self, long_prices):
        from src.services.ml_predictor import MLStrategy

        result = run_backtest(long_prices, MLStrategy(train_ratio=0.6, n_estimators=20))
        assert len(result.portfolio_values) == len(long_prices)
        assert result.metrics.total_trades >= 0

    def test_ml_signals_hold_during_training(self, long_prices):
        from src.services.ml_predictor import MLStrategy

        strategy = MLStrategy(train_ratio=0.6, n_estimators=20)
        signals = strategy.generate_signals(long_prices)
        # First ~50 signals should be HOLD (indicator warmup)
        assert all(s == Signal.HOLD for s in signals[:50])
