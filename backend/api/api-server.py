from fastapi import FastAPI
from multiprocessing import Pool
import torch
from app.config import APIConfig
from app.utils import initialize_quantum_pool
import uvicorn

class QuantumServer:
    def __init__(self):
        self.config = APIConfig()
        self.quantum_pool = None
        
    def initialize_quantum_workers(self):
        self.quantum_pool = Pool(
            processes=self.config.NUM_WORKERS,
            initializer=initialize_quantum_pool,
            initargs=(self.config.FIELD_STRENGTH, self.config.REALITY_COHERENCE)
        )
        
    def start(self):
        self.initialize_quantum_workers()
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            workers=self.config.NUM_WORKERS
        )
        
    def shutdown(self):
        if self.quantum_pool:
            self.quantum_pool.close()
            self.quantum_pool.join()

if __name__ == "__main__":
    server = QuantumServer()
    server.start()
