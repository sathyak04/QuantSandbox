"""Celery tasks for running simulations and backtests asynchronously."""

import logging
import uuid

import numpy as np

from src.database import SessionLocal
from src.models.backtest import BacktestRun, BacktestStatus, Trade, TradeAction
from src.models.simulation import PricePath, Simulation, SimulationStatus
from src.services.backtester import run_backtest
from src.services.simulator import SimulationParams, simulate_gbm
from src.services.strategies import Signal, get_strategy_class
from src.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="run_simulation", bind=True, max_retries=2)
def run_simulation(self, simulation_id: str) -> dict:
    """Generate price paths for a simulation using GBM Monte Carlo."""
    db = SessionLocal()
    try:
        sim = db.get(Simulation, uuid.UUID(simulation_id))
        if not sim:
            return {"error": "Simulation not found"}

        sim.status = SimulationStatus.RUNNING
        db.commit()

        params = SimulationParams(
            initial_price=sim.initial_price,
            drift=sim.drift,
            volatility=sim.volatility,
            time_steps=sim.time_steps,
            num_paths=sim.num_paths,
        )

        prices = simulate_gbm(params)

        for i in range(params.num_paths):
            path = PricePath(
                simulation_id=sim.id,
                path_index=i,
                prices=prices[i].tolist(),
            )
            db.add(path)

        sim.status = SimulationStatus.COMPLETED
        db.commit()

        logger.info("Simulation %s completed: %d paths generated", simulation_id, params.num_paths)
        return {"status": "completed", "num_paths": params.num_paths}

    except Exception as exc:
        db.rollback()
        if sim:
            sim.status = SimulationStatus.FAILED
            db.commit()
        logger.exception("Simulation %s failed", simulation_id)
        raise self.retry(exc=exc, countdown=5)
    finally:
        db.close()


@celery_app.task(name="run_backtest", bind=True, max_retries=2)
def run_backtest_task(self, backtest_id: str) -> dict:
    """Run a trading strategy backtest on a simulation's price path."""
    db = SessionLocal()
    try:
        bt = db.get(BacktestRun, uuid.UUID(backtest_id))
        if not bt:
            return {"error": "Backtest not found"}

        bt.status = BacktestStatus.RUNNING
        db.commit()

        # Get the price path
        path_index = bt.strategy_params.get("path_index", 0)
        price_path = (
            db.query(PricePath)
            .filter(
                PricePath.simulation_id == bt.simulation_id,
                PricePath.path_index == path_index,
            )
            .first()
        )

        if not price_path:
            bt.status = BacktestStatus.FAILED
            db.commit()
            return {"error": "Price path not found"}

        prices = np.array(price_path.prices)

        # Instantiate the strategy
        try:
            strategy_cls = get_strategy_class(bt.strategy_type.value)
        except ValueError:
            bt.status = BacktestStatus.FAILED
            db.commit()
            return {"error": f"Unknown strategy: {bt.strategy_type}"}

        # Filter out non-strategy params before constructing
        reserved_keys = {"initial_cash", "path_index"}
        strategy_params = {k: v for k, v in bt.strategy_params.items() if k not in reserved_keys}
        strategy = strategy_cls(**strategy_params)

        initial_cash = bt.strategy_params.get("initial_cash", 100_000.0)
        result = run_backtest(prices, strategy, initial_cash=initial_cash)

        # Persist results
        bt.total_return = result.metrics.total_return
        bt.sharpe_ratio = result.metrics.sharpe_ratio
        bt.max_drawdown = result.metrics.max_drawdown
        bt.win_count = result.metrics.win_count
        bt.loss_count = result.metrics.loss_count
        bt.portfolio_values = result.portfolio_values

        for trade in result.trades:
            db.add(Trade(
                backtest_id=bt.id,
                step=trade.step,
                action=TradeAction.BUY if trade.action == Signal.BUY else TradeAction.SELL,
                price=trade.price,
                quantity=trade.quantity,
                pnl=trade.pnl,
            ))

        bt.status = BacktestStatus.COMPLETED
        db.commit()

        logger.info(
            "Backtest %s completed: return=%.2f%%, sharpe=%.2f",
            backtest_id,
            result.metrics.total_return * 100,
            result.metrics.sharpe_ratio,
        )
        return {
            "status": "completed",
            "total_return": result.metrics.total_return,
            "sharpe_ratio": result.metrics.sharpe_ratio,
        }

    except Exception as exc:
        db.rollback()
        if bt:
            bt.status = BacktestStatus.FAILED
            db.commit()
        logger.exception("Backtest %s failed", backtest_id)
        raise self.retry(exc=exc, countdown=5)
    finally:
        db.close()
