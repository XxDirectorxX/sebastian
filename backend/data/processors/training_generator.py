import torch
import numpy as np
from pathlib import Path
import pickle
import json

class TrainingDataGenerator:
    def __init__(self):
        self.base_path = Path("R:/Sebastian-Rebuild/backend/data")
        self.training_path = self.base_path / "training"
        self.batch_size = 2048
        self.field_strength = 46.97871376
        
    def generate_training_data(self):
        # Load intents
        with open(self.base_path / "intents.json", 'r') as f:
            intents = json.load(f)
            
        # Load cleaned data
        df = pd.read_csv(self.base_path / "cleaned-data.csv")
        
        # Generate training tensors
        X = self._generate_input_tensors(df['input'].values)
        y = self._generate_target_tensors(df['intent'].values)
        
        # Save training data
        training_data = {
            'X': X.numpy(),
            'y': y.numpy(),
            'intents': intents
        }
        
        with open(self.training_path / "training_data.pkl", 'wb') as f:
            pickle.dump(training_data, f)
            
    def _generate_input_tensors(self, texts: np.ndarray) -> torch.Tensor:
        # Convert text to tensors
        tensors = []
        for text in texts:
            tensor = self._text_to_tensor(text)
            tensors.append(tensor)
        return torch.stack(tensors)
        
    def _generate_target_tensors(self, intents: np.ndarray) -> torch.Tensor:
        # Convert intents to one-hot encoded tensors
        unique_intents = np.unique(intents)
        intent_map = {intent: i for i, intent in enumerate(unique_intents)}
        return torch.tensor([intent_map[i] for i in intents])
