import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.database import get_db
from src.models.simulation import Simulation, SimulationStatus
from src.schemas.simulation import SimulationCreate, SimulationDetail, SimulationResponse
from src.worker.tasks import run_simulation

router = APIRouter()


@router.post("/", response_model=SimulationResponse, status_code=status.HTTP_202_ACCEPTED)
def create_simulation(payload: SimulationCreate, db: Session = Depends(get_db)):
    sim = Simulation(
        name=payload.name,
        source=payload.source,
        ticker=payload.ticker,
        initial_price=payload.initial_price,
        drift=payload.drift,
        volatility=payload.volatility,
        time_steps=payload.time_steps,
        num_paths=payload.num_paths,
        status=SimulationStatus.PENDING,
    )
    db.add(sim)
    db.commit()
    db.refresh(sim)

    run_simulation.delay(str(sim.id))
    return sim


@router.get("/", response_model=list[SimulationResponse])
def list_simulations(db: Session = Depends(get_db)):
    return db.query(Simulation).order_by(Simulation.created_at.desc()).all()


@router.get("/{simulation_id}", response_model=SimulationDetail)
def get_simulation(simulation_id: uuid.UUID, db: Session = Depends(get_db)):
    sim = db.get(Simulation, simulation_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return sim


@router.delete("/{simulation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_simulation(simulation_id: uuid.UUID, db: Session = Depends(get_db)):
    sim = db.get(Simulation, simulation_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    db.delete(sim)
    db.commit()
