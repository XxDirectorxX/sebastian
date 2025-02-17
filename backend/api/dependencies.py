from backend.config import settings
from import_manager import *

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_quantum_processor():
    processor = torch.zeros(
        (APIConfig.MATRIX_DIMENSION, 
         APIConfig.MATRIX_DIMENSION, 
         APIConfig.MATRIX_DIMENSION),
        dtype=torch.complex128,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    return processor

def get_reality_field():
    return torch.exp(1j * APIConfig.REALITY_COHERENCE ** APIConfig.UPDATE_RATE)
def verify_quantum_token(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.TOKEN_ALGORITHM]
        )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid quantum authentication"
        )

def get_current_reality(payload: dict = Depends(verify_quantum_token)):
    return payload.get("reality_id")
