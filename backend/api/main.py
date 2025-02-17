from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, Optional
import asyncio
import torch
from pathlib import Path

from core.quantum_field_manager import QuantumFieldManager, QuantumFieldConfig
from core.field_stability_controller import FieldStabilityController, StabilityConfig
from integration.field_integration_service import FieldIntegrationService, IntegrationConfig
from integration.websocket_manager import WebSocketManager, WebSocketConfig
from integration.resource_controller import ResourceController, ResourceConfig

app = FastAPI(title="Sebastian Quantum API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize quantum systems
quantum_manager = QuantumFieldManager()
stability_controller = FieldStabilityController()
websocket_manager = WebSocketManager()
resource_controller = ResourceController(ResourceConfig())

@app.on_event("startup")
async def startup_event():
    # Initialize quantum field
    await quantum_manager.initialize_quantum_field()
    
    # Start resource monitoring
    asyncio.create_task(resource_controller.monitor_resources())

@app.on_event("shutdown")
async def shutdown_event():
    await websocket_manager.cleanup()
    await resource_controller.cleanup()

@app.websocket("/ws/quantum")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            quantum_state = torch.tensor(data["quantum_state"])
            processed_state = await quantum_manager.process_quantum_field(quantum_state)
            await websocket_manager.broadcast_quantum_state(processed_state)
    except WebSocketDisconnect:
        await websocket_manager.disconnect(websocket)

@app.get("/metrics")
async def get_metrics():
    return {
        "quantum_metrics": quantum_manager.get_metrics(),
        "stability_metrics": stability_controller.get_metrics(),
        "resource_metrics": resource_controller.get_resource_metrics()
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)