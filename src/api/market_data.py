from datetime import date

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.services.market_data import fetch_historical_prices

router = APIRouter()


class MarketDataResponse(BaseModel):
    ticker: str
    prices: list[float]
    dates: list[str]
    drift: float
    volatility: float
    num_points: int


@router.get("/{ticker}", response_model=MarketDataResponse)
def get_market_data(
    ticker: str,
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
    period: str = Query(default="1y"),
):
    try:
        result = fetch_historical_prices(ticker, start=start, end=end, period=period)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return MarketDataResponse(
        ticker=result.ticker,
        prices=result.prices,
        dates=result.dates,
        drift=round(result.drift, 6),
        volatility=round(result.volatility, 6),
        num_points=len(result.prices),
    )
