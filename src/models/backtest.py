import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class StrategyType(str, enum.Enum):
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    RANDOM = "random"
    ML_PREDICTED = "ml_predicted"


class TradeAction(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class BacktestStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    simulation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("simulations.id", ondelete="CASCADE")
    )
    strategy_type: Mapped[StrategyType] = mapped_column(Enum(StrategyType))
    strategy_params: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[BacktestStatus] = mapped_column(
        Enum(BacktestStatus), default=BacktestStatus.PENDING
    )
    total_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    win_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    loss_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    portfolio_values: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    simulation: Mapped["Simulation"] = relationship(back_populates="backtest_runs")  # noqa: F821
    trades: Mapped[list["Trade"]] = relationship(
        back_populates="backtest_run", cascade="all, delete-orphan"
    )


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    backtest_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("backtest_runs.id", ondelete="CASCADE")
    )
    step: Mapped[int] = mapped_column(Integer)
    action: Mapped[TradeAction] = mapped_column(Enum(TradeAction))
    price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)

    backtest_run: Mapped["BacktestRun"] = relationship(back_populates="trades")
