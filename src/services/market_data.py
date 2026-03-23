"""Fetch real historical market data via yfinance."""

from dataclasses import dataclass
from datetime import date

import numpy as np
import yfinance as yf


@dataclass
class MarketDataResult:
    ticker: str
    prices: list[float]
    dates: list[str]
    drift: float  # annualized mean return
    volatility: float  # annualized std of returns


def fetch_historical_prices(
    ticker: str,
    start: date | None = None,
    end: date | None = None,
    period: str = "1y",
) -> MarketDataResult:
    """Download adjusted close prices from Yahoo Finance.

    Args:
        ticker: Stock symbol (e.g. "AAPL", "MSFT")
        start/end: Date range. If None, uses `period` instead.
        period: yfinance period string (e.g. "1y", "6mo", "5y")

    Returns:
        MarketDataResult with prices, dates, and computed stats.

    Raises:
        ValueError: If ticker returns no data.
    """
    stock = yf.Ticker(ticker)

    if start and end:
        df = stock.history(start=start.isoformat(), end=end.isoformat())
    else:
        df = stock.history(period=period)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    prices = df["Close"].values.astype(float)
    dates = [d.strftime("%Y-%m-%d") for d in df.index]

    # Compute annualized stats from daily log returns
    log_returns = np.diff(np.log(prices))
    drift = float(np.mean(log_returns) * 252)
    volatility = float(np.std(log_returns) * np.sqrt(252))

    return MarketDataResult(
        ticker=ticker.upper(),
        prices=prices.tolist(),
        dates=dates,
        drift=drift,
        volatility=volatility,
    )
