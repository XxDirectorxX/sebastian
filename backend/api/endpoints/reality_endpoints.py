from fastapi import APIRouter, WebSocket, Depends
from app.reality_field import SOTARealityField, RealityFieldConfig
import torch
from typing import Dict, List
from dataclasses import dataclass

router = APIRouter()

@dataclass
class RealityRequest:
    quantum_state: List[float]
    field_strength: float = 46.97871376
    reality_coherence: float = 1.618033988749895

@dataclass
class RealityResponse:
    processed_state: List[float]
    field_metrics: Dict[str, float]
    coherence_status: str

class RealityFieldEndpoints:
    def __init__(self):
        self.config = RealityFieldConfig()
        self.reality_field = SOTARealityField(self.config)
        
    @router.post("/reality/process")
    async def process_reality(self, request: RealityRequest):
        quantum_state = torch.tensor(request.quantum_state)
        processed_state = await self.reality_field.process_reality_field(
            quantum_state,
            request.field_strength
        )
        
        metrics = self.reality_field._calculate_field_metrics(processed_state)
        
        return RealityResponse(
            processed_state=processed_state.tolist(),
            field_metrics=metrics,
            coherence_status=self._get_coherence_status(metrics)
        )

    @router.websocket("/reality/stream")
    async def reality_stream(self, websocket: WebSocket):
        await self.reality_field.stream_reality_field(
            websocket,
            self._get_quantum_processor()
        )

    def _get_coherence_status(self, metrics: Dict[str, float]) -> str:
        coherence = metrics["reality_coherence"]
        if coherence > 0.9:
            return "OPTIMAL"
        elif coherence > 0.7:
            return "STABLE"
        return "DEGRADED"

reality_endpoints = RealityFieldEndpoints()
