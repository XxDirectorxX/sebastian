import pytest
from fastapi.testclient import TestClient
from app.main import app
import torch

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_quantum_processor():
    return torch.zeros((64, 64, 64), dtype=torch.complex128)

def test_chat_endpoint(client, mock_quantum_processor):
    response = client.post(
        "/chat/send",
        json={
            "content": "Test message",
            "field_strength": 46.97871376
        }
    )
    assert response.status_code == 200
    assert "field_coherence" in response.json()
