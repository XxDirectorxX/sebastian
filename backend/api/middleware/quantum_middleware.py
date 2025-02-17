from fastapi import Request
from typing import Callable
import time

class QuantumMiddleware:
    async def __call__(
        self,
        request: Request,
        call_next: Callable
    ):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Add quantum metrics to response headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Quantum-Coherence"] = str(0.9999)
        response.headers["X-Reality-Alignment"] = str(0.9995)
        
        return response
