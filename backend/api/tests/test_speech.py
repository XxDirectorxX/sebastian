import pytest
from fastapi.testclient import TestClient
from app.main import app
import torch
import numpy as np

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_audio_data():
    return np.random.randn(16000).tolist()

def test_transcribe_endpoint(client, mock_audio_data):
    response = client.post(
        "/speech/transcribe",
        json={
            "audio_data": mock_audio_data,
            "sample_rate": 16000,
            "field_strength": 46.97871376
        }
    )
    assert response.status_code == 200
    assert "field_coherence" in response.json()
    assert "reality_alignment" in response.json()
