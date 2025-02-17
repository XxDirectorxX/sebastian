from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Optional
import torch
from ..core.quantum_field_manager import QuantumFieldManager
from ..core.field_stability_controller import FieldStabilityController

router = APIRouter()
quantum_manager = QuantumFieldManager()
stability_controller = FieldStabilityController()

@router.post("/process")
async def process_quantum_state(state: Dict[str, list]):
    quantum_state = torch.tensor(state["data"])
    processed_state, metrics = await quantum_manager.process_quantum_field(quantum_state)
    return {"state": processed_state.tolist(), "metrics": metrics}

@router.post("/stabilize")
async def stabilize_quantum_state(state: Dict[str, list]):
    quantum_state = torch.tensor(state["data"])
    stabilized_state, metrics = await stability_controller.maintain_field_stability(quantum_state)
    return {"state": stabilized_state.tolist(), "metrics": metrics}

@router.get("/metrics")
async def get_quantum_metrics():
    return {
        "quantum_metrics": quantum_manager.get_metrics(),
        "stability_metrics": stability_controller.get_metrics()
    }
