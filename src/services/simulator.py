"""Stock price simulation engine using Geometric Brownian Motion."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationParams:
    initial_price: float
    drift: float  # annualized expected return (mu)
    volatility: float  # annualized standard deviation (sigma)
    time_steps: int  # number of discrete time periods
    num_paths: int = 100  # Monte Carlo paths
    dt: float = 1 / 252  # time increment (default: 1 trading day)


def simulate_gbm(params: SimulationParams, seed: int | None = None) -> np.ndarray:
    """Simulate stock prices using Geometric Brownian Motion.

    The GBM model:
        dS = mu * S * dt + sigma * S * dW

    Discretized (Euler-Maruyama):
        S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    Returns:
        np.ndarray of shape (num_paths, time_steps + 1) with price paths.
        Column 0 is the initial price for all paths.
    """
    rng = np.random.default_rng(seed)

    z = rng.standard_normal((params.num_paths, params.time_steps))

    log_returns = (
        (params.drift - 0.5 * params.volatility**2) * params.dt
        + params.volatility * np.sqrt(params.dt) * z
    )

    # Cumulative sum of log returns, prepend 0 for initial price
    cum_log_returns = np.concatenate(
        [np.zeros((params.num_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
    )

    prices = params.initial_price * np.exp(cum_log_returns)
    return prices


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from a price series."""
    return np.diff(np.log(prices), axis=-1)


def compute_simple_returns(prices: np.ndarray) -> np.ndarray:
    """Compute simple returns from a price series."""
    return np.diff(prices, axis=-1) / prices[..., :-1]
