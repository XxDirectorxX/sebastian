from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
from typing import Optional
import jwt
from datetime import datetime, timedelta

class QuantumSecurity:
    SECRET_KEY = "quantum_reality_key_46.97871376"
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    api_key_header = APIKeyHeader(name="X-Quantum-Key", auto_error=True)

    @staticmethod
    async def create_access_token(data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=QuantumSecurity.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, QuantumSecurity.SECRET_KEY, algorithm=QuantumSecurity.ALGORITHM)

    @staticmethod
    async def verify_token(token: str = Security(api_key_header)) -> Optional[dict]:
        try:
            payload = jwt.decode(token, QuantumSecurity.SECRET_KEY, algorithms=[QuantumSecurity.ALGORITHM])
            return payload
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid quantum authentication token")

security = QuantumSecurity()
