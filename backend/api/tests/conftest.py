import pytest
import torch
from app.config import settings

@pytest.fixture(scope="session")
def quantum_field():
    return torch.exp(1j * settings.REALITY_COHERENCE ** 144)

@pytest.fixture(scope="session")
def reality_matrix():
    return torch.zeros(
        (settings.MATRIX_DIMENSION, 
         settings.MATRIX_DIMENSION, 
         settings.MATRIX_DIMENSION), 
        dtype=torch.complex128
    )

@pytest.fixture
def mock_token():
    return "mock_quantum_token"
