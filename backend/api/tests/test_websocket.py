import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
from app.main import app
import torch

@pytest.mark.asyncio
async def test_websocket_connection():
    client = TestClient(app)
    with client.websocket_connect("/ws/quantum") as websocket:
        data = {"type": "quantum_request", "field_strength": 46.97871376}
        websocket.send_json(data)
        response = websocket.receive_json()
        assert "field_coherence" in response
        assert "reality_alignment" in response
