"""Core quantum framework initialization"""
from .processor import Processor
from .operator import Operator
from .tensor import Tensor
from .field import Field
from .state import State
from .logger import QuantumLogger

# Framework constants
FIELD_STRENGTH = 46.97871376
REALITY_COHERENCE = 1.618033988749895
QUANTUM_TENSOR_SIZE = (64, 64, 64)
REALITY_FIELD_SIZE = (31, 31, 31)

__all__ = [
    'Processor',
    'Operator',
    'Tensor',
    'Field',
    'State',
    'QuantumLogger',
    'FIELD_STRENGTH',
    'REALITY_COHERENCE',
    'QUANTUM_TENSOR_SIZE',
    'REALITY_FIELD_SIZE'
]