"""ML-based trading strategy using Random Forest on technical indicators."""

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.services.strategies import Signal


@dataclass
class TechnicalIndicators:
    """Computed feature matrix from a price series."""

    sma_10: np.ndarray
    sma_50: np.ndarray
    rsi_14: np.ndarray
    macd: np.ndarray
    macd_signal: np.ndarray
    bollinger_upper: np.ndarray
    bollinger_lower: np.ndarray
    atr_14: np.ndarray


def compute_sma(prices: np.ndarray, window: int) -> np.ndarray:
    """Simple Moving Average."""
    out = np.full_like(prices, np.nan)
    for i in range(window - 1, len(prices)):
        out[i] = np.mean(prices[i - window + 1 : i + 1])
    return out


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    deltas = np.diff(prices)
    rsi = np.full(len(prices), np.nan)

    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def compute_macd(
    prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray]:
    """MACD line and signal line using EMA."""

    def ema(data: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1)
        out = np.full_like(data, np.nan)
        out[span - 1] = np.mean(data[:span])
        for i in range(span, len(data)):
            out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
        return out

    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow

    # Signal line is EMA of MACD
    valid_start = slow - 1
    macd_signal = np.full_like(macd_line, np.nan)
    macd_vals = macd_line[valid_start:]
    sig = ema(macd_vals, signal)
    macd_signal[valid_start:] = sig

    return macd_line, macd_signal


def compute_bollinger_bands(
    prices: np.ndarray, window: int = 20, num_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Bollinger Bands (upper, lower)."""
    sma = compute_sma(prices, window)
    std = np.full_like(prices, np.nan)
    for i in range(window - 1, len(prices)):
        std[i] = np.std(prices[i - window + 1 : i + 1])
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower


def compute_atr(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range (approximated from close prices only)."""
    tr = np.abs(np.diff(prices))
    atr = np.full(len(prices), np.nan)
    if len(tr) >= period:
        atr[period] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i + 1] = (atr[i] * (period - 1) + tr[i]) / period
    return atr


def build_features(prices: np.ndarray) -> tuple[np.ndarray, int]:
    """Build feature matrix from price array.

    Returns:
        (features, valid_start_index) — features has shape (n_valid, n_features),
        valid_start_index is the first index in the original prices array that
        has a complete feature row.
    """
    sma_10 = compute_sma(prices, 10)
    sma_50 = compute_sma(prices, 50)
    rsi = compute_rsi(prices, 14)
    macd_line, macd_signal = compute_macd(prices)
    bb_upper, bb_lower = compute_bollinger_bands(prices, 20)
    atr = compute_atr(prices, 14)

    # Stack all features
    all_features = np.column_stack([
        prices,
        sma_10,
        sma_50,
        rsi,
        macd_line,
        macd_signal,
        bb_upper,
        bb_lower,
        atr,
        (prices - sma_10) / sma_10,  # price deviation from SMA10
        (prices - sma_50) / sma_50,  # price deviation from SMA50
    ])

    # Find first row with no NaNs
    valid_mask = ~np.any(np.isnan(all_features), axis=1)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        raise ValueError("Not enough data to compute all indicators (need at least 50 data points)")

    valid_start = valid_indices[0]
    features = all_features[valid_start:]

    return features, valid_start


def _create_labels(prices: np.ndarray, threshold: float = 0.001) -> np.ndarray:
    """Create classification labels: 1 (buy) if next return > threshold, 0 (sell) otherwise.

    Uses next-day return as the target.
    """
    returns = np.diff(prices) / prices[:-1]
    labels = (returns > threshold).astype(int)
    return labels


@dataclass
class MLStrategy:
    """ML-based strategy: trains a Random Forest on technical indicators."""

    train_ratio: float = 0.6
    n_estimators: int = 100
    threshold: float = 0.001
    seed: int = 42

    _model: RandomForestClassifier = field(init=False, repr=False, default=None)
    _scaler: StandardScaler = field(init=False, repr=False, default=None)
    _train_end: int = field(init=False, repr=False, default=0)

    @property
    def name(self) -> str:
        return "ml_predicted"

    @property
    def params(self) -> dict:
        return {
            "train_ratio": self.train_ratio,
            "n_estimators": self.n_estimators,
            "threshold": self.threshold,
        }

    def generate_signals(self, prices: np.ndarray) -> list[Signal]:
        """Train on first `train_ratio` of data, predict on the rest.

        Returns signals for ALL time steps (HOLD during training period).
        """
        features, valid_start = build_features(prices)
        valid_prices = prices[valid_start:]

        # Labels: does the next step go up?
        labels = _create_labels(valid_prices, self.threshold)

        # Features and labels must align: features[:-1] predicts labels
        X = features[:-1]
        y = labels

        train_size = int(len(X) * self.train_ratio)
        if train_size < 20:
            raise ValueError("Not enough training data for ML strategy")

        X_train, X_test = X[:train_size], X[train_size:]
        y_train = y[:train_size]

        # Scale features
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        # Train
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            n_jobs=-1,
        )
        self._model.fit(X_train_scaled, y_train)

        # Predict on test set
        predictions = self._model.predict(X_test_scaled)

        # Build full signal list
        signals: list[Signal] = []

        # Before valid_start: HOLD (not enough indicator data)
        signals.extend([Signal.HOLD] * valid_start)

        # Training period: HOLD
        signals.extend([Signal.HOLD] * (train_size + 1))

        # Test period: use predictions
        for pred in predictions:
            signals.append(Signal.BUY if pred == 1 else Signal.SELL)

        # Pad if needed (last step has no label)
        while len(signals) < len(prices):
            signals.append(Signal.HOLD)

        self._train_end = valid_start + train_size + 1
        return signals[: len(prices)]
