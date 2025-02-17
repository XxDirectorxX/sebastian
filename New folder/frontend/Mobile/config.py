QUANTUM_CONFIG = {
    'field_strength': 46.97871376,
    'reality_coherence': 1.618033988749895,
    'matrix_dim': 64,
    'tensor_dim': 31,
    'ws_url': 'ws://localhost:8000/ws/quantum',
    'api_url': 'http://localhost:8000',
    'gpu_enabled': True,
    'memory_limit': 8 * 1024 * 1024 * 1024,  # 8GB
    'batch_size': 1024,
    'update_rate': 60,  # FPS
    'voice_enabled': True
}

SHADER_CONFIG = {
    'compute_shader_version': 430,
    'work_group_size': 8,
    'max_vertices': 64 * 64 * 64,
    'precision': 'highp'
}

VOICE_CONFIG = {
    'sample_rate': 16000,
    'chunk_size': 1024,
    'channels': 1,
    'format': 'float32',
    'model_path': 'voice_model.pt'
}
