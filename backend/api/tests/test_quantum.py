import pytest
import torch
from ..core.quantum_field_manager import QuantumFieldManager
from ..core.field_stability_controller import FieldStabilityController

@pytest.fixture
def quantum_manager():
    return QuantumFieldManager()

@pytest.fixture
def stability_controller():
    return FieldStabilityController()

async def test_quantum_field_processing(quantum_manager):
    # Test quantum field processing
    test_state = torch.randn(64, 64, 64, dtype=torch.complex128)
    processed_state, metrics = await quantum_manager.process_quantum_field(test_state)
    
    assert processed_state.shape == (64, 64, 64)
    assert metrics["field_strength"] > 0
    assert 0 <= metrics["quantum_coherence"] <= 1
    assert -np.pi <= metrics["reality_alignment"] <= np.pi

async def test_field_stability(stability_controller):
    # Test field stability maintenance
    test_state = torch.randn(64, 64, 64, dtype=torch.complex128)
    stabilized_state, metrics = await stability_controller.maintain_field_stability(test_state)
    
    assert stabilized_state.shape == (64, 64, 64)
    assert metrics["coherence_level"] > 0.9
    assert metrics["stability_factor"] > 0.8
