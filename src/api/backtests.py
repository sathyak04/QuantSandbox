import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from src.database import get_db
from src.models.backtest import BacktestRun, BacktestStatus
from src.schemas.backtest import (
    BacktestComparison,
    BacktestCreate,
    BacktestDetail,
    BacktestResponse,
)
from src.worker.tasks import run_backtest_task

router = APIRouter()


@router.post("/", response_model=BacktestResponse, status_code=status.HTTP_202_ACCEPTED)
def create_backtest(payload: BacktestCreate, db: Session = Depends(get_db)):
    bt = BacktestRun(
        simulation_id=payload.simulation_id,
        strategy_type=payload.strategy_type,
        strategy_params={
            **payload.strategy_params,
            "initial_cash": payload.initial_cash,
            "path_index": payload.path_index,
        },
        status=BacktestStatus.PENDING,
    )
    db.add(bt)
    db.commit()
    db.refresh(bt)

    run_backtest_task.delay(str(bt.id))
    return bt


@router.get("/", response_model=list[BacktestResponse])
def list_backtests(
    simulation_id: uuid.UUID | None = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(BacktestRun).order_by(BacktestRun.created_at.desc())
    if simulation_id:
        query = query.filter(BacktestRun.simulation_id == simulation_id)
    return query.all()


@router.get("/compare", response_model=BacktestComparison)
def compare_backtests(
    ids: list[uuid.UUID] = Query(..., alias="id"),
    db: Session = Depends(get_db),
):
    results = db.query(BacktestRun).filter(BacktestRun.id.in_(ids)).all()
    return BacktestComparison(backtests=results)


@router.get("/{backtest_id}", response_model=BacktestDetail)
def get_backtest(backtest_id: uuid.UUID, db: Session = Depends(get_db)):
    bt = db.get(BacktestRun, backtest_id)
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return bt
