import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class SimulationSource(str, enum.Enum):
    SYNTHETIC = "synthetic"
    HISTORICAL = "historical"


class SimulationStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Simulation(Base):
    __tablename__ = "simulations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255))
    source: Mapped[SimulationSource] = mapped_column(Enum(SimulationSource))
    ticker: Mapped[str | None] = mapped_column(String(20), nullable=True)
    initial_price: Mapped[float] = mapped_column(Float)
    drift: Mapped[float] = mapped_column(Float)
    volatility: Mapped[float] = mapped_column(Float)
    time_steps: Mapped[int] = mapped_column(Integer)
    num_paths: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[SimulationStatus] = mapped_column(
        Enum(SimulationStatus), default=SimulationStatus.PENDING
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    price_paths: Mapped[list["PricePath"]] = relationship(
        back_populates="simulation", cascade="all, delete-orphan"
    )
    backtest_runs = relationship(
        "BacktestRun", back_populates="simulation", cascade="all, delete-orphan"
    )


class PricePath(Base):
    __tablename__ = "price_paths"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    simulation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("simulations.id", ondelete="CASCADE")
    )
    path_index: Mapped[int] = mapped_column(Integer)
    prices: Mapped[list] = mapped_column(JSONB)

    simulation: Mapped["Simulation"] = relationship(back_populates="price_paths")
