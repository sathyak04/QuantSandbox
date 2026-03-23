"""Trading strategy implementations.

All strategies implement the same Protocol so they're interchangeable
in the backtesting framework.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np


class Signal(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradeRecord:
    step: int
    action: Signal
    price: float
    quantity: float
    pnl: float | None = None  # realized P&L, set on sells


class TradingStrategy(Protocol):
    """Interface that all strategies must implement."""

    @property
    def name(self) -> str: ...

    @property
    def params(self) -> dict: ...

    def generate_signals(self, prices: np.ndarray) -> list[Signal]:
        """Given a price array, return a signal for each time step."""
        ...


@dataclass
class MeanReversionStrategy:
    """Buy when price drops X% below moving average, sell when Y% above."""

    window: int = 20
    buy_threshold: float = -0.02  # buy when price is 2% below MA
    sell_threshold: float = 0.02  # sell when price is 2% above MA

    @property
    def name(self) -> str:
        return "mean_reversion"

    @property
    def params(self) -> dict:
        return {
            "window": self.window,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
        }

    def generate_signals(self, prices: np.ndarray) -> list[Signal]:
        signals = []
        for i in range(len(prices)):
            if i < self.window:
                signals.append(Signal.HOLD)
                continue

            ma = np.mean(prices[i - self.window : i])
            deviation = (prices[i] - ma) / ma

            if deviation <= self.buy_threshold:
                signals.append(Signal.BUY)
            elif deviation >= self.sell_threshold:
                signals.append(Signal.SELL)
            else:
                signals.append(Signal.HOLD)

        return signals


@dataclass
class MomentumStrategy:
    """Buy on strong upward momentum, sell on downward momentum."""

    lookback: int = 10
    threshold: float = 0.03  # 3% move over lookback period

    @property
    def name(self) -> str:
        return "momentum"

    @property
    def params(self) -> dict:
        return {"lookback": self.lookback, "threshold": self.threshold}

    def generate_signals(self, prices: np.ndarray) -> list[Signal]:
        signals = []
        for i in range(len(prices)):
            if i < self.lookback:
                signals.append(Signal.HOLD)
                continue

            price_change = (prices[i] - prices[i - self.lookback]) / prices[i - self.lookback]

            if price_change >= self.threshold:
                signals.append(Signal.BUY)
            elif price_change <= -self.threshold:
                signals.append(Signal.SELL)
            else:
                signals.append(Signal.HOLD)

        return signals


@dataclass
class RandomStrategy:
    """Random buy/sell baseline for comparison."""

    buy_probability: float = 0.05
    sell_probability: float = 0.05
    seed: int | None = 42

    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    @property
    def name(self) -> str:
        return "random"

    @property
    def params(self) -> dict:
        return {
            "buy_probability": self.buy_probability,
            "sell_probability": self.sell_probability,
            "seed": self.seed,
        }

    def generate_signals(self, prices: np.ndarray) -> list[Signal]:
        signals = []
        rolls = self._rng.random(len(prices))
        for roll in rolls:
            if roll < self.buy_probability:
                signals.append(Signal.BUY)
            elif roll < self.buy_probability + self.sell_probability:
                signals.append(Signal.SELL)
            else:
                signals.append(Signal.HOLD)
        return signals


def _get_strategy_registry() -> dict[str, type]:
    """Lazy import to avoid circular deps with ml_predictor."""
    from src.services.ml_predictor import MLStrategy

    return {
        "mean_reversion": MeanReversionStrategy,
        "momentum": MomentumStrategy,
        "random": RandomStrategy,
        "ml_predicted": MLStrategy,
    }


STRATEGY_REGISTRY: dict[str, type[TradingStrategy]] = {
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
    "random": RandomStrategy,
}


def get_strategy_class(name: str) -> type:
    registry = _get_strategy_registry()
    if name not in registry:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(registry.keys())}")
    return registry[name]
