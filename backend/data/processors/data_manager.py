from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import Dict, Any

class DataManager:
    def __init__(self):
        self.base_path = Path("R:/Sebastian-Rebuild/backend/data")
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.training_path = self.base_path / "training"
        
    def load_intents(self) -> Dict[str, Any]:
        with open(self.base_path / "intents.json", 'r') as f:
            return json.load(f)
            
    def load_speech_config(self) -> Dict[str, Any]:
        with open(self.base_path / "speech-config.json", 'r') as f:
            return json.load(f)
            
    def load_cleaned_data(self) -> pd.DataFrame:
        return pd.read_csv(self.base_path / "cleaned-data.csv")
        
    def load_training_data(self) -> Dict[str, np.ndarray]:
        with open(self.base_path / "training-data.pkl", 'rb') as f:
            return pickle.load(f)
            
    def save_processed_data(self, data: pd.DataFrame, filename: str):
        data.to_csv(self.processed_path / filename, index=False)
        
    def save_training_checkpoint(self, checkpoint: Dict[str, Any], filename: str):
        with open(self.training_path / filename, 'wb') as f:
            pickle.dump(checkpoint, f)
