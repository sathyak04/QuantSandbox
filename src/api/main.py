from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.backtests import router as backtests_router
from src.api.market_data import router as market_data_router
from src.api.simulations import router as simulations_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="QuantSandbox",
    description="Trading simulation sandbox with backtesting and ML predictions",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulations_router, prefix="/api/simulations", tags=["simulations"])
app.include_router(backtests_router, prefix="/api/backtests", tags=["backtests"])
app.include_router(market_data_router, prefix="/api/market-data", tags=["market-data"])


@app.get("/health")
def health_check():
    return {"status": "ok"}
