from import_manager import *
from datetime import datetime
from typing import Optional, Dict, Any, Union
import logging
import sys
from pathlib import Path
import json
import queue
from concurrent.futures import ThreadPoolExecutor
import os

class QuantumLogger:
    """Advanced Quantum Framework Logger for monitoring, debugging and metrics tracking"""
    
    def __init__(self, name: str, log_dir: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory
        self.log_dir = log_dir or Path("R:/sebastian/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Multiple log handlers
        self._setup_file_handlers()
        self._setup_console_handler()
        self._setup_quantum_metrics_handler()
        
        # Performance monitoring
        self.start_time = time.time()
        self.operation_counter = 0
        self.error_counter = 0
        
        # Metrics queue for async processing
        self.metrics_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize metrics storage
        self.quantum_metrics = {}
        self.performance_data = []
        self.error_logs = []

    def _setup_file_handlers(self):
        """Setup multiple specialized file handlers"""
        # Main log file
        main_handler = logging.FileHandler(self.log_dir / "quantum_framework.log")
        main_handler.setLevel(logging.INFO)
        
        # Error log file
        error_handler = logging.FileHandler(self.log_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        
        # Performance log file
        perf_handler = logging.FileHandler(self.log_dir / "performance.log")
        perf_handler.setLevel(logging.INFO)
        
        # Metrics log file
        metrics_handler = logging.FileHandler(self.log_dir / "metrics.log")
        metrics_handler.setLevel(logging.INFO)
        
        # Set formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        for handler in [main_handler, error_handler, perf_handler, metrics_handler]:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _setup_console_handler(self):
        """Setup console output handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _setup_quantum_metrics_handler(self):
        """Setup quantum metrics handler"""
        self.metrics_handler = logging.FileHandler(self.log_dir / "quantum_metrics.json")
        self.metrics_handler.setLevel(logging.INFO)

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log info message with optional context"""
        self.operation_counter += 1
        self.logger.info(f"{msg} | Operations: {self.operation_counter}", extra=kwargs)
        self._async_process_metrics(kwargs)

    def error(self, msg: str, error: Exception = None, **kwargs: Any) -> None:
        """Log error with full context and stacktrace"""
        self.error_counter += 1
        error_msg = f"{msg} | Error: {str(error) if error else 'Unknown'}"
        self.logger.error(error_msg, exc_info=True, extra=kwargs)
        self.error_logs.append({
            'timestamp': datetime.now().isoformat(),
            'message': error_msg,
            'context': kwargs
        })

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log warning message with context"""
        self.logger.warning(msg, extra=kwargs)

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log debug message with context"""
        self.logger.debug(msg, extra=kwargs)

    def quantum_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log quantum operation metrics asynchronously"""
        self.metrics_queue.put(('quantum', metrics))
        self.executor.submit(self._process_quantum_metrics, metrics)

    def performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        metrics = {
            'elapsed_time': time.time() - self.start_time,
            'total_operations': self.operation_counter,
            'error_count': self.error_counter,
            'error_rate': self.error_counter / max(1, self.operation_counter),
            'timestamp': datetime.now().isoformat()
        }
        self.performance_data.append(metrics)
        return metrics

    def state_change(self, state: str, details: Dict[str, Any]) -> None:
        """Log quantum state changes with validation"""
        self.logger.info(f"State Change: {state}", extra={'details': details})
        self._validate_and_store_state(state, details)

    def field_update(self, field_type: str, metrics: Dict[str, Any]) -> None:
        """Log field updates with metrics"""
        self.logger.info(f"Field Update: {field_type}", extra={'metrics': metrics})
        self._store_field_metrics(field_type, metrics)

    def _async_process_metrics(self, metrics: Dict[str, Any]) -> None:
        """Process metrics asynchronously"""
        self.executor.submit(self._process_metrics, metrics)

    def _process_quantum_metrics(self, metrics: Dict[str, Any]) -> None:
        """Process and store quantum metrics"""
        timestamp = datetime.now().isoformat()
        self.quantum_metrics[timestamp] = metrics
        self._save_metrics_to_file()

    def _validate_and_store_state(self, state: str, details: Dict[str, Any]) -> None:
        """Validate and store quantum state information"""
        if self._validate_state(details):
            self.logger.info(f"Valid state change: {state}")
            self._store_state(state, details)
        else:
            self.error(f"Invalid state: {state}", details=details)

    def _validate_state(self, details: Dict[str, Any]) -> bool:
        """Validate quantum state details"""
        required_keys = ['quantum_state', 'coherence', 'stability']
        return all(key in details for key in required_keys)

    def _store_state(self, state: str, details: Dict[str, Any]) -> None:
        """Store quantum state information"""
        with open(self.log_dir / "states.json", 'a') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'state': state,
                'details': details
            }, f)
            f.write('\n')

    def _store_field_metrics(self, field_type: str, metrics: Dict[str, Any]) -> None:
        """Store field metrics"""
        with open(self.log_dir / "field_metrics.json", 'a') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'field_type': field_type,
                'metrics': metrics
            }, f)
            f.write('\n')

    def _save_metrics_to_file(self) -> None:
        """Save accumulated metrics to file"""
        with open(self.log_dir / "quantum_metrics.json", 'w') as f:
            json.dump(self.quantum_metrics, f, indent=2)

    def export_metrics(self, format: str = 'json') -> Union[str, Dict[str, Any]]:
        """Export all metrics in specified format"""
        metrics = {
            'quantum_metrics': self.quantum_metrics,
            'performance_data': self.performance_data,
            'error_logs': self.error_logs,
            'summary': self.performance_metrics()
        }
        
        if format == 'json':
            return json.dumps(metrics, indent=2)
        return metrics

    def cleanup(self) -> None:
        """Cleanup logger resources"""
        self.executor.shutdown(wait=True)
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        self._save_metrics_to_file()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()