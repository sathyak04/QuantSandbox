# QuantSandbox

[![CI](https://github.com/sathyak04/QuantSandbox/actions/workflows/ci.yml/badge.svg)](https://github.com/sathyak04/QuantSandbox/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sathyak04/QuantSandbox/graph/badge.svg)](https://codecov.io/gh/sathyak04/QuantSandbox)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED.svg?logo=docker&logoColor=white)](docker-compose.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A trading simulation platform that generates synthetic or real stock price data, runs multiple trading strategies, and evaluates performance through backtesting — all served via a FastAPI REST API with async job processing and an interactive Streamlit dashboard.

## Architecture

```
Streamlit Dashboard → FastAPI REST API → Service Layer → PostgreSQL
                                       ↘ Celery + Redis (async jobs)
```

## Features

- **Price Simulation** — Geometric Brownian Motion Monte Carlo engine with configurable drift, volatility, and path count
- **Real Market Data** — Yahoo Finance integration for historical prices on any ticker
- **4 Trading Strategies** — Mean Reversion, Momentum, Random baseline, and ML-based (Random Forest on technical indicators)
- **Backtesting Engine** — Computes Sharpe ratio, max drawdown, total return, win/loss counts, and full equity curves
- **ML Prediction Layer** — Random Forest classifier trained on 8 technical indicators (SMA, RSI, MACD, Bollinger Bands, ATR)
- **Async Job Processing** — Celery + Redis for running simulations and backtests as background tasks
- **REST API** — FastAPI with proper async patterns (202 Accepted → poll for results)
- **Interactive Dashboard** — Streamlit UI with price path visualization, trade signal overlays, and strategy comparison charts

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI, Pydantic, Uvicorn |
| Database | PostgreSQL, SQLAlchemy 2.0, Alembic |
| Async Jobs | Celery, Redis |
| ML | scikit-learn, NumPy, Pandas |
| Dashboard | Streamlit, Plotly |
| Infrastructure | Docker Compose, GitHub Actions CI |

## Quick Start

### Full Stack (Docker)

```bash
cp .env.example .env
docker compose up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |

### Dashboard Only (no Docker needed)

```bash
pip install -e .
streamlit run dashboard/app.py
```

## API Endpoints

```
POST   /api/simulations          Create & queue a new simulation (202)
GET    /api/simulations          List all simulations
GET    /api/simulations/{id}     Get simulation detail + price paths
DELETE /api/simulations/{id}     Delete a simulation

POST   /api/backtests            Run strategy backtest on a simulation (202)
GET    /api/backtests            List backtest results
GET    /api/backtests/{id}       Get backtest detail + trades
GET    /api/backtests/compare    Compare metrics across backtests

GET    /api/market-data/{ticker} Fetch real historical data via Yahoo Finance
```

## Trading Strategies

| Strategy | Logic | Key Parameters |
|----------|-------|---------------|
| **Mean Reversion** | Buy when price drops below moving average, sell when above | `window`, `buy_threshold`, `sell_threshold` |
| **Momentum** | Buy on strong upward movement, sell on downward | `lookback`, `threshold` |
| **Random** | Random buy/sell as baseline comparison | `buy_probability`, `sell_probability` |
| **ML (Random Forest)** | Train on technical indicators, predict next-day direction | `train_ratio`, `n_estimators` |

## Project Structure

```
├── src/
│   ├── api/              FastAPI routes
│   ├── models/           SQLAlchemy ORM models
│   ├── schemas/          Pydantic request/response schemas
│   ├── services/         Core business logic (simulator, strategies, backtester, ML)
│   └── worker/           Celery async tasks
├── dashboard/            Streamlit frontend
├── tests/                Pytest test suite
├── alembic/              Database migrations
├── docker-compose.yml    Full stack orchestration
└── .github/workflows/    CI pipeline
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/
```

## License

[MIT](LICENSE)
