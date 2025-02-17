import torch
from pathlib import Path
import json
import pickle

class ModelSerializer:
    def __init__(self):
        self.base_path = Path("R:/Sebastian-Rebuild/backend/data")
        self.training_path = self.base_path / "training"
        self.field_strength = 46.97871376
        
    def save_model(self, model: torch.nn.Module, metadata: dict):
        # Save model state
        torch.save(model.state_dict(), 
                  self.training_path / "model_state.pt")
        
        # Save model metadata
        with open(self.training_path / "model_meta.json", 'w') as f:
            json.dump(metadata, f, indent=4)
            
    def load_model(self, model_class: torch.nn.Module):
        # Load model state
        state_dict = torch.load(self.training_path / "model_state.pt")
        
        # Load metadata
        with open(self.training_path / "model_meta.json", 'r') as f:
            metadata = json.load(f)
            
        # Initialize and load model
        model = model_class(**metadata['model_params'])
        model.load_state_dict(state_dict)
        
        return model, metadata
