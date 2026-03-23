import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from src.models.simulation import SimulationSource, SimulationStatus


class SimulationCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    source: SimulationSource = SimulationSource.SYNTHETIC
    ticker: str | None = Field(default=None, max_length=20)
    initial_price: float = Field(gt=0, default=100.0)
    drift: float = Field(default=0.05, description="Annualized expected return")
    volatility: float = Field(gt=0, default=0.2, description="Annualized volatility")
    time_steps: int = Field(gt=0, le=10000, default=252)
    num_paths: int = Field(gt=0, le=1000, default=100)


class PricePathResponse(BaseModel):
    id: uuid.UUID
    path_index: int
    prices: list[float]

    model_config = {"from_attributes": True}


class SimulationResponse(BaseModel):
    id: uuid.UUID
    name: str
    source: SimulationSource
    ticker: str | None
    initial_price: float
    drift: float
    volatility: float
    time_steps: int
    num_paths: int
    status: SimulationStatus
    created_at: datetime

    model_config = {"from_attributes": True}


class SimulationDetail(SimulationResponse):
    price_paths: list[PricePathResponse] = []
