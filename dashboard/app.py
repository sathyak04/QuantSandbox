"""QuantSandbox — Interactive Trading Simulation Dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.services.backtester import run_backtest
from src.services.market_data import fetch_historical_prices
from src.services.ml_predictor import MLStrategy
from src.services.simulator import SimulationParams, simulate_gbm
from src.services.strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    RandomStrategy,
    Signal,
)

st.set_page_config(page_title="QuantSandbox", layout="wide")
st.title("QuantSandbox — Trading Simulation Platform")

# ─── Sidebar: Data Source ─────────────────────────────────────────────────────
st.sidebar.header("1. Data Source")
data_source = st.sidebar.radio("Price data", ["Synthetic (GBM)", "Historical (Yahoo Finance)"])

if data_source == "Synthetic (GBM)":
    st.sidebar.subheader("GBM Parameters")
    initial_price = st.sidebar.number_input("Initial Price ($)", value=100.0, min_value=0.01)
    drift = st.sidebar.slider("Drift (annualized return)", -0.5, 1.0, 0.05, 0.01)
    volatility = st.sidebar.slider("Volatility (annualized)", 0.01, 1.0, 0.20, 0.01)
    time_steps = st.sidebar.slider("Time Steps (trading days)", 50, 2000, 252)
    num_paths = st.sidebar.slider("Monte Carlo Paths", 1, 200, 10)
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0)

    params = SimulationParams(
        initial_price=initial_price,
        drift=drift,
        volatility=volatility,
        time_steps=time_steps,
        num_paths=num_paths,
    )
    all_prices = simulate_gbm(params, seed=int(seed))
    dates = None
else:
    st.sidebar.subheader("Historical Data")
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
    period = st.sidebar.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)

    if ticker:
        try:
            result = fetch_historical_prices(ticker, period=period)
            all_prices = np.array([result.prices])
            dates = result.dates
            num_paths = 1
            st.sidebar.success(
                f"{result.ticker}: {len(result.prices)} days | "
                f"drift={result.drift:.2%} | vol={result.volatility:.2%}"
            )
        except ValueError as e:
            st.sidebar.error(str(e))
            st.stop()
    else:
        st.stop()

# ─── Tab Layout ───────────────────────────────────────────────────────────────
tab_sim, tab_backtest, tab_compare = st.tabs(
    ["Price Simulation", "Strategy Backtest", "Compare Strategies"]
)

# ─── Tab 1: Price Paths ──────────────────────────────────────────────────────
with tab_sim:
    st.header("Simulated Price Paths")

    fig = go.Figure()
    x_axis = dates if dates else list(range(all_prices.shape[1]))

    for i in range(all_prices.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=all_prices[i],
                mode="lines",
                name=f"Path {i}",
                opacity=0.6,
            )
        )

    fig.update_layout(
        xaxis_title="Date" if dates else "Time Step",
        yaxis_title="Price ($)",
        height=500,
        showlegend=all_prices.shape[0] <= 20,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    final_prices = all_prices[:, -1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Final Price", f"${np.mean(final_prices):.2f}")
    col2.metric("Std Dev", f"${np.std(final_prices):.2f}")
    col3.metric("Min", f"${np.min(final_prices):.2f}")
    col4.metric("Max", f"${np.max(final_prices):.2f}")

# ─── Tab 2: Single Strategy Backtest ─────────────────────────────────────────
with tab_backtest:
    st.header("Strategy Backtest")

    col_strat, col_params = st.columns([1, 2])

    with col_strat:
        path_idx = st.selectbox(
            "Price Path",
            range(all_prices.shape[0]),
            format_func=lambda x: f"Path {x}",
        )
        strategy_name = st.selectbox(
            "Strategy", ["Mean Reversion", "Momentum", "Random", "ML (Random Forest)"]
        )
        initial_cash = st.number_input("Initial Cash ($)", value=100_000.0, min_value=1.0)

    with col_params:
        if strategy_name == "Mean Reversion":
            window = st.slider("MA Window", 5, 100, 20)
            buy_thresh = st.slider("Buy Threshold (%)", -10.0, 0.0, -2.0, 0.5) / 100
            sell_thresh = st.slider("Sell Threshold (%)", 0.0, 10.0, 2.0, 0.5) / 100
            strategy = MeanReversionStrategy(window=window, buy_threshold=buy_thresh, sell_threshold=sell_thresh)
        elif strategy_name == "Momentum":
            lookback = st.slider("Lookback Period", 3, 50, 10)
            threshold = st.slider("Momentum Threshold (%)", 0.5, 10.0, 3.0, 0.5) / 100
            strategy = MomentumStrategy(lookback=lookback, threshold=threshold)
        elif strategy_name == "Random":
            buy_prob = st.slider("Buy Probability", 0.01, 0.20, 0.05, 0.01)
            sell_prob = st.slider("Sell Probability", 0.01, 0.20, 0.05, 0.01)
            strategy = RandomStrategy(buy_probability=buy_prob, sell_probability=sell_prob)
        else:
            train_ratio = st.slider("Train Ratio", 0.3, 0.8, 0.6, 0.05)
            n_estimators = st.slider("Number of Trees", 10, 500, 100, 10)
            strategy = MLStrategy(train_ratio=train_ratio, n_estimators=n_estimators)

    prices = all_prices[path_idx]

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            result = run_backtest(prices, strategy, initial_cash=initial_cash)

        # Metrics row
        m = result.metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Return", f"{m.total_return:+.2%}")
        c2.metric("Sharpe Ratio", f"{m.sharpe_ratio:.2f}")
        c3.metric("Max Drawdown", f"{m.max_drawdown:.2%}")
        c4.metric("Wins", m.win_count)
        c5.metric("Losses", m.loss_count)

        # Price + signals chart
        x_axis = dates if dates else list(range(len(prices)))

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
            subplot_titles=["Price & Trade Signals", "Portfolio Value"],
        )

        fig.add_trace(
            go.Scatter(x=x_axis, y=prices, mode="lines", name="Price", line=dict(color="#636EFA")),
            row=1, col=1,
        )

        # Buy/sell markers
        buy_steps = [t.step for t in result.trades if t.action == Signal.BUY]
        sell_steps = [t.step for t in result.trades if t.action == Signal.SELL]

        buy_x = [x_axis[s] for s in buy_steps if s < len(x_axis)]
        buy_y = [prices[s] for s in buy_steps if s < len(prices)]
        sell_x = [x_axis[s] for s in sell_steps if s < len(x_axis)]
        sell_y = [prices[s] for s in sell_steps if s < len(prices)]

        fig.add_trace(
            go.Scatter(x=buy_x, y=buy_y, mode="markers", name="Buy",
                       marker=dict(symbol="triangle-up", size=12, color="#00CC96")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=sell_x, y=sell_y, mode="markers", name="Sell",
                       marker=dict(symbol="triangle-down", size=12, color="#EF553B")),
            row=1, col=1,
        )

        # Portfolio value
        fig.add_trace(
            go.Scatter(x=x_axis, y=result.portfolio_values, mode="lines",
                       name="Portfolio", line=dict(color="#AB63FA")),
            row=2, col=1,
        )

        fig.update_layout(height=700, showlegend=True)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Value ($)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Trade log
        if result.trades:
            st.subheader("Trade Log")
            trade_data = [
                {
                    "Step": t.step,
                    "Action": t.action.value,
                    "Price": f"${t.price:.2f}",
                    "Quantity": f"{t.quantity:.4f}",
                    "P&L": f"${t.pnl:.2f}" if t.pnl is not None else "—",
                }
                for t in result.trades
            ]
            st.dataframe(pd.DataFrame(trade_data), use_container_width=True)

# ─── Tab 3: Compare All Strategies ───────────────────────────────────────────
with tab_compare:
    st.header("Strategy Comparison")

    compare_path = st.selectbox(
        "Price Path for Comparison",
        range(all_prices.shape[0]),
        format_func=lambda x: f"Path {x}",
        key="compare_path",
    )
    compare_cash = st.number_input("Initial Cash ($)", value=100_000.0, min_value=1.0, key="compare_cash")

    if st.button("Compare All Strategies", type="primary"):
        prices = all_prices[compare_path]

        strategies = [
            MeanReversionStrategy(),
            MomentumStrategy(),
            RandomStrategy(),
        ]
        # Only add ML if enough data points
        if len(prices) >= 100:
            strategies.append(MLStrategy())

        results = {}
        for strat in strategies:
            with st.spinner(f"Running {strat.name}..."):
                results[strat.name] = run_backtest(prices, strat, initial_cash=compare_cash)

        # Metrics comparison table
        metrics_data = []
        for name, res in results.items():
            m = res.metrics
            metrics_data.append({
                "Strategy": name,
                "Total Return": f"{m.total_return:+.2%}",
                "Sharpe Ratio": f"{m.sharpe_ratio:.2f}",
                "Max Drawdown": f"{m.max_drawdown:.2%}",
                "Wins": m.win_count,
                "Losses": m.loss_count,
                "Total Trades": m.total_trades,
            })
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

        # Sharpe ratio bar chart
        fig_sharpe = go.Figure(
            data=[
                go.Bar(
                    x=list(results.keys()),
                    y=[r.metrics.sharpe_ratio for r in results.values()],
                    marker_color=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"][: len(results)],
                )
            ]
        )
        fig_sharpe.update_layout(title="Sharpe Ratio Comparison", yaxis_title="Sharpe Ratio", height=350)
        st.plotly_chart(fig_sharpe, use_container_width=True)

        # Portfolio growth overlay
        x_axis = dates if dates else list(range(len(prices)))
        fig_portfolio = go.Figure()
        colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

        for i, (name, res) in enumerate(results.items()):
            fig_portfolio.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=res.portfolio_values,
                    mode="lines",
                    name=name,
                    line=dict(color=colors[i % len(colors)]),
                )
            )

        fig_portfolio.update_layout(
            title="Portfolio Growth Comparison",
            xaxis_title="Date" if dates else "Time Step",
            yaxis_title="Portfolio Value ($)",
            height=450,
        )
        st.plotly_chart(fig_portfolio, use_container_width=True)
