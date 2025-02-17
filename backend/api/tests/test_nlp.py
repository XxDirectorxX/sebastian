import pytest
from fastapi.testclient import TestClient
from app.main import app
import torch

@pytest.fixture
def client():
    return TestClient(app)

def test_nlp_process_endpoint(client):
    response = client.post(
        "/nlp/process",
        json={
            "text": "Test quantum processing",
            "task": "analysis",
            "field_strength": 46.97871376,
            "reality_coherence": 1.618033988749895
        }
    )
    assert response.status_code == 200
    assert "quantum_state" in response.json()
    assert "field_metrics" in response.json()
