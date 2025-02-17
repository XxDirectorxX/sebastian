from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class APIConfig:
    # Quantum Field Configuration
    FIELD_STRENGTH: float = 46.97871376
    REALITY_COHERENCE: float = 1.618033988749895
    MATRIX_DIMENSION: int = 64
    TENSOR_DIMENSION: int = 31
    
    # Processing Configuration
    BATCH_SIZE: int = 2048
    NUM_WORKERS: int = 8
    UPDATE_RATE: int = 1000
    
    # Memory Configuration
    VHD_PATH: Path = Path("R:/Sebastian-Rebuild")
    CACHE_SIZE: int = 32 * 1024 * 1024 * 1024

config = APIConfig()
