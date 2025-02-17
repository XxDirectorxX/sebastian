from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path

# Quantum Constants
FIELD_STRENGTH = 46.97871376
REALITY_COHERENCE = 1.618033988749895
QUANTUM_SPEED = 0.0001
TENSOR_ALIGNMENT = 0.99999

@dataclass 
class QuantumConfig:
    num_qubits: int = 8
    shots: int = 1024
    backend: str = "qasm_simulator"
    error_correction: bool = True
    batch_size: int = 32
    use_half_precision: bool = True
    cache_size: int = 1000
    gpu_memory_fraction: float = 0.95
    tensor_parallel: bool = True
    enable_cuda_graphs: bool = True
    mixed_precision: bool = True
    field_strength: float = FIELD_STRENGTH
    reality_coherence: float = REALITY_COHERENCE
    tensor_alignment: float = TENSOR_ALIGNMENT
    quantum_speed: float = QUANTUM_SPEED

@dataclass
class ProcessorConfig:
    field_processor: bool = True
    tensor_processor: bool = True
    operator_processor: bool = True
    state_processor: bool = True
    quantum_processor: bool = True
    emotion_processor: bool = True
    personality_processor: bool = True
    unified_processor: bool = True
    voice_processor: bool = True

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    gradient_accumulation: int = 1
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

@dataclass
class NLPConfig:
    model_name: str = "meta-llama/Llama-2-70b-chat-hf"
    max_length: int = 512
    batch_size: int = 32
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    
@dataclass
class SpeechConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    f_min: float = 0.0
    f_max: float = 8000.0
    power: float = 1.0

@dataclass
class ChatConfig:
    model_name: str = "meta-llama/Llama-2-70b-chat-hf"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    personality: str = "sebastian"
    response_temp: float = 0.7