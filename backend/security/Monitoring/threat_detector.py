import torch
import numpy as np
from typing import Optional, Tuple

class ThreatDetector:
    def __init__(self):
        self.field_strength = 46.97871376
        self.reality_coherence = 1.618033988749895
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def detect_threats(self, input_state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        state = input_state.to(self.device)
        threat_field = self._generate_threat_field(state)
        threat_score = self._calculate_threat_level(threat_field)
        return threat_field, threat_score

    def _generate_threat_field(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.exp(1j * self.field_strength * tensor)

    def _calculate_threat_level(self, tensor: torch.Tensor) -> float:
        return float(torch.mean(torch.abs(tensor)))
