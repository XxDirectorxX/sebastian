from fastapi import APIRouter, Depends, WebSocket, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.dependencies import get_quantum_processor, get_current_user
import torch
from app.config import settings

router = APIRouter()

class ChatMessage(BaseModel):
    content: str
    role: str = "user"
    field_strength: Optional[float] = settings.FIELD_STRENGTH

class ChatResponse(BaseModel):
    content: str
    field_coherence: float
    reality_matrix: List[float]

@router.post("/send", response_model=ChatResponse)
async def process_message(
    message: ChatMessage,
    quantum_processor: torch.Tensor = Depends(get_quantum_processor),
    current_user: str = Depends(get_current_user)
):
    # Initialize quantum field for chat processing
    field = torch.exp(1j * settings.REALITY_COHERENCE ** 144)
    coherence = field * message.field_strength
    
    # Process message through quantum field
    response_tensor = quantum_processor * coherence
    
    # Generate response using quantum state
    response = {
        "content": "Processed quantum response",
        "field_coherence": float(torch.abs(coherence).mean()),
        "reality_matrix": response_tensor.flatten().tolist()[:10]
    }
    
    return response

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    quantum_processor: torch.Tensor = Depends(get_quantum_processor)
):
    await websocket.accept()
    
    while True:
        message = await websocket.receive_text()
        field = torch.exp(1j * settings.REALITY_COHERENCE ** 144)
        response = await process_quantum_message(message, field, quantum_processor)
        await websocket.send_json(response)
