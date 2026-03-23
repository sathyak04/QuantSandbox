"""Backtesting framework that evaluates strategies on price data."""

from dataclasses import dataclass

import numpy as np

from src.services.strategies import Signal, TradeRecord, TradingStrategy


@dataclass
class BacktestMetrics:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_count: int
    loss_count: int
    total_trades: int


@dataclass
class BacktestResult:
    metrics: BacktestMetrics
    trades: list[TradeRecord]
    portfolio_values: list[float]
    signals: list[Signal]


def run_backtest(
    prices: np.ndarray,
    strategy: TradingStrategy,
    initial_cash: float = 100_000.0,
) -> BacktestResult:
    """Run a strategy against a single price path.

    Simple position model:
    - BUY: go all-in (spend all cash on shares)
    - SELL: liquidate entire position
    - HOLD: do nothing

    Tracks portfolio value (cash + holdings) at each step.
    """
    signals = strategy.generate_signals(prices)

    cash = initial_cash
    shares = 0.0
    entry_price = 0.0
    trades: list[TradeRecord] = []
    portfolio_values: list[float] = []

    for i, (price, signal) in enumerate(zip(prices, signals)):
        # Execute trades
        if signal == Signal.BUY and cash > 0:
            shares = cash / price
            entry_price = price
            trades.append(TradeRecord(step=i, action=signal, price=price, quantity=shares))
            cash = 0.0

        elif signal == Signal.SELL and shares > 0:
            cash = shares * price
            pnl = (price - entry_price) * shares
            trades.append(
                TradeRecord(step=i, action=signal, price=price, quantity=shares, pnl=pnl)
            )
            shares = 0.0

        # Record portfolio value
        portfolio_values.append(cash + shares * price)

    metrics = _compute_metrics(portfolio_values, trades, initial_cash)
    return BacktestResult(
        metrics=metrics,
        trades=trades,
        portfolio_values=portfolio_values,
        signals=signals,
    )


def _compute_metrics(
    portfolio_values: list[float],
    trades: list[TradeRecord],
    initial_cash: float,
) -> BacktestMetrics:
    """Compute performance metrics from backtest results."""
    pv = np.array(portfolio_values)

    # Total return
    total_return = (pv[-1] - initial_cash) / initial_cash

    # Daily returns for Sharpe ratio
    daily_returns = np.diff(pv) / pv[:-1]
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe_ratio = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
    else:
        sharpe_ratio = 0.0

    # Maximum drawdown
    peak = np.maximum.accumulate(pv)
    drawdowns = (pv - peak) / peak
    max_drawdown = float(np.min(drawdowns))

    # Win/loss counts (only on sell trades with realized P&L)
    sell_trades = [t for t in trades if t.action == Signal.SELL and t.pnl is not None]
    win_count = sum(1 for t in sell_trades if t.pnl > 0)
    loss_count = sum(1 for t in sell_trades if t.pnl <= 0)

    return BacktestMetrics(
        total_return=round(total_return, 6),
        sharpe_ratio=round(sharpe_ratio, 4),
        max_drawdown=round(max_drawdown, 6),
        win_count=win_count,
        loss_count=loss_count,
        total_trades=len(trades),
    )
