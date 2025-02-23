"""Quantum Framework Configuration"""
import torch
import numpy as np
from pathlib import Path
import logging
import sys

# Version
VERSION = "1.0.0"

# Constants
QUANTUM_TENSOR_SIZE = (64, 64, 64)
REALITY_FIELD_SIZE = (31, 31, 31)
COHERENCE_MATRIX_SIZE = (128, 128, 128)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_AVAILABLE = torch.cuda.is_available()

# Framework constants
FIELD_STRENGTH = 46.97871376
REALITY_COHERENCE = 1.618033988749895
QUANTUM_CHANNELS = 8
OPTIMIZATION_LEVEL = 3

# Directory structure
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / 'logs'
MODELS_DIR = BASE_DIR / 'models'
CACHE_DIR = BASE_DIR / 'cache'
CONFIG_DIR = BASE_DIR / 'config'

# Create directories
for dir_path in [LOGS_DIR, MODELS_DIR, CACHE_DIR, CONFIG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "quantum_framework.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Quantum Framework v{VERSION} initialized on {DEVICE}")

# Framework configuration
CONFIG = {
    'version': VERSION,
    'device': str(DEVICE),
    'cuda_available': CUDA_AVAILABLE,
    'field_strength': FIELD_STRENGTH,
    'reality_coherence': REALITY_COHERENCE,
    'quantum_channels': QUANTUM_CHANNELS,
    'optimization_level': OPTIMIZATION_LEVEL,
    'tensor_size': QUANTUM_TENSOR_SIZE,
    'field_size': REALITY_FIELD_SIZE,
    'matrix_size': COHERENCE_MATRIX_SIZE
}

# Import core components
from .core import *
from .processing import *
from .processors import *
from .integration import *
from .optimization import *
from .orchestration import *
from .reality import *
from .stabilization import *
from .validation import *

__all__ = [
    'Processor',
    'Field',
    'Tensor',
    'QuantumLogger',
    'State',
    'Operator',
    'CONFIG',
    'VERSION',
    'DEVICE'
]