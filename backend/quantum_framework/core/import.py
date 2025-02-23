import os
from pathlib import Path

core_files = {
    "__init__.py": {
        "required_elements": ["initialize_framework", "DEFAULT_CONFIG", "__version__"],
        "size": "~100 lines"
    },
    "processor.py": {
        "required_elements": ["Processor", "process_quantum_state", "_initialize_quantum_circuit"],
        "size": "~400 lines"
    },
    "operator.py": {
        "required_elements": ["Operator", "apply_operator", "_initialize_quantum_components"],
        "size": "~500 lines"
    },
    "tensor.py": {
        "required_elements": ["Tensor", "process_tensor", "_initialize_tensor_fields"],
        "size": "~400 lines"
    },
    "field.py": {
        "required_elements": ["Field", "process_field", "_initialize_field_components"],
        "size": "~400 lines"
    },
    "state.py": {
        "required_elements": ["State", "evolve_quantum_state", "_initialize_quantum_system"],
        "size": "~400 lines"
    },
    "logger.py": {
        "required_elements": ["QuantumLogger", "quantum_metrics", "_setup_quantum_metrics_handler"],
        "size": "~200 lines"
    }
}

core_path = Path("R:/sebastian/backend/quantum_framework/core")