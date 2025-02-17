from fastapi import APIRouter, WebSocket, BackgroundTasks
from app.quantum_monitor import QuantumFieldMonitor, MonitoringConfig
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass
import asyncio
from collections import deque

router = APIRouter()

@dataclass
class MonitoringResponse:
    quantum_metrics: Dict[str, float]
    system_health: Dict[str, str]
    performance_analysis: Dict[str, float]
    resource_utilization: Dict[str, float]

class QuantumMonitoringEndpoints:
    def __init__(self):
        self.config = MonitoringConfig()
        self.monitor = QuantumFieldMonitor(self.config)
        self.metric_buffer = deque(maxlen=10000)
        
    @router.get("/monitor/metrics")
    async def get_quantum_metrics(self):
        quantum_state = self._get_current_quantum_state()
        metrics = await self.monitor.monitor_quantum_operations(
            quantum_state,
            field_strength=46.97871376
        )
        
        analysis = await self.monitor.get_performance_analysis()
        
        return MonitoringResponse(
            quantum_metrics=metrics,
            system_health=self._analyze_system_health(metrics),
            performance_analysis=analysis,
            resource_utilization=self._get_resource_metrics()
        )

    @router.websocket("/monitor/stream")
    async def monitor_stream(self, websocket: WebSocket):
        await websocket.accept()
        
        try:
            while True:
                quantum_state = self._get_current_quantum_state()
                metrics = await self.monitor.monitor_quantum_operations(
                    quantum_state,
                    field_strength=46.97871376
                )
                
                self.metric_buffer.append(metrics)
                
                await websocket.send_json({
                    "current_metrics": metrics,
                    "trend_analysis": self._analyze_trends(),
                    "system_status": self._get_system_status()
                })
                
                await asyncio.sleep(1 / self.config.sampling_rate)
                
        except Exception as e:
            await websocket.close()

    def _get_current_quantum_state(self) -> torch.Tensor:
        return torch.randn(
            (self.config.buffer_size, 64, 64),
            device=self.monitor.device,
            dtype=torch.complex128
        )

    def _analyze_system_health(self, metrics: Dict[str, float]) -> Dict[str, str]:
        return {
            "quantum_coherence": "OPTIMAL" if metrics["quantum_coherence"] > 0.95 else "DEGRADED",
            "reality_alignment": "STABLE" if metrics["reality_alignment"] > 0.9 else "UNSTABLE",
            "processing_efficiency": "HIGH" if metrics["processing_latency"] < 1000000 else "LOW"
        }

    def _get_resource_metrics(self) -> Dict[str, float]:
        return {
            "vhd_usage": self.monitor._get_storage_metrics()["used_space"],
            "gpu_memory": self.monitor._get_gpu_metrics()["memory_used"],
            "gpu_utilization": self.monitor._get_gpu_metrics()["utilization"]
        }

    def _analyze_trends(self) -> Dict[str, float]:
        recent_metrics = list(self.metric_buffer)[-1000:]
        return {
            "coherence_trend": self._calculate_trend([m["quantum_coherence"] for m in recent_metrics]),
            "alignment_trend": self._calculate_trend([m["reality_alignment"] for m in recent_metrics]),
            "latency_trend": self._calculate_trend([m["processing_latency"] for m in recent_metrics])
        }

    def _calculate_trend(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return (values[-1] - values[0]) / len(values)

    def _get_system_status(self) -> Dict[str, str]:
        return {
            "quantum_processor": "ONLINE",
            "reality_field": "STABLE",
            "monitoring_system": "ACTIVE",
            "metric_collection": "OPERATIONAL"
        }

quantum_monitor_endpoints = QuantumMonitoringEndpoints()
