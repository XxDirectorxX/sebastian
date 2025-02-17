from fastapi.testclient import TestClient
from ..main import app
import torch
import numpy as np

client = TestClient(app)

def test_quantum_process_endpoint():
    test_data = {
        "data": torch.randn(64, 64, 64).tolist(),
        "field_strength": 46.97871376
    }
    response = client.post("/quantum/process", json=test_data)
    assert response.status_code == 200
    assert "state" in response.json()
    assert "metrics" in response.json()

def test_quantum_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "quantum_metrics" in response.json()
    assert "stability_metrics" in response.json()
