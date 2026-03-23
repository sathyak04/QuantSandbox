import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from src.models.backtest import BacktestStatus, StrategyType, TradeAction


class BacktestCreate(BaseModel):
    simulation_id: uuid.UUID
    strategy_type: StrategyType
    strategy_params: dict = Field(default_factory=dict)
    initial_cash: float = Field(gt=0, default=100_000.0)
    path_index: int = Field(ge=0, default=0, description="Which price path to backtest on")


class TradeResponse(BaseModel):
    id: uuid.UUID
    step: int
    action: TradeAction
    price: float
    quantity: float
    pnl: float | None

    model_config = {"from_attributes": True}


class BacktestResponse(BaseModel):
    id: uuid.UUID
    simulation_id: uuid.UUID
    strategy_type: StrategyType
    strategy_params: dict
    status: BacktestStatus
    total_return: float | None
    sharpe_ratio: float | None
    max_drawdown: float | None
    win_count: int | None
    loss_count: int | None
    created_at: datetime

    model_config = {"from_attributes": True}


class BacktestDetail(BacktestResponse):
    portfolio_values: list[float] | None = None
    trades: list[TradeResponse] = []


class BacktestComparison(BaseModel):
    backtests: list[BacktestResponse]
