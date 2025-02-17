from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from app.dependencies import get_quantum_processor, get_current_user
from app.config import settings

router = APIRouter()

class NLPRequest(BaseModel):
    text: str
    task: str
    field_strength: Optional[float] = settings.FIELD_STRENGTH
    reality_coherence: Optional[float] = settings.REALITY_COHERENCE

class NLPResponse(BaseModel):
    result: str
    confidence: float
    field_metrics: dict
    quantum_state: List[float]

@router.post("/process", response_model=NLPResponse)
async def process_text(
    request: NLPRequest,
    quantum_processor: torch.Tensor = Depends(get_quantum_processor),
    current_user: str = Depends(get_current_user)
):
    # Initialize quantum field for NLP
    field = torch.exp(1j * request.reality_coherence ** 144)
    coherence = field * request.field_strength
    
    # Convert text to quantum tensor
    text_tensor = torch.tensor([ord(c) for c in request.text], dtype=torch.float32)
    quantum_text = text_tensor * coherence
    
    # Process through quantum field
    processed_tensor = quantum_processor * quantum_text
    
    field_metrics = {
        "field_strength": float(torch.abs(coherence).mean()),
        "reality_coherence": float(request.reality_coherence),
        "quantum_alignment": float(torch.abs(processed_tensor).std())
    }
    
    return {
        "result": "Quantum processed text",
        "confidence": float(torch.abs(coherence).mean()),
        "field_metrics": field_metrics,
        "quantum_state": processed_tensor.flatten().tolist()[:10]
    }

@router.post("/analyze", response_model=NLPResponse)
async def analyze_text(
    request: NLPRequest,
    quantum_processor: torch.Tensor = Depends(get_quantum_processor),
    current_user: str = Depends(get_current_user)
):
    # Quantum field initialization for analysis
    field = torch.exp(1j * request.reality_coherence ** 233)
    coherence = field * request.field_strength
    
    # Process text through quantum analysis
    text_tensor = torch.tensor([ord(c) for c in request.text], dtype=torch.float32)
    analyzed_tensor = quantum_processor * (text_tensor * coherence)
    
    return {
        "result": "Quantum text analysis",
        "confidence": float(torch.abs(coherence).mean()),
        "field_metrics": {
            "field_strength": float(request.field_strength),
            "coherence": float(torch.abs(coherence).mean()),
            "stability": float(torch.abs(analyzed_tensor).std())
        },
        "quantum_state": analyzed_tensor.flatten().tolist()[:10]
    }
