import requests
import os
from pathlib import Path
import shutil

def recover_framework_directories():
    # Framework directories to recover
    directories = {
        "core": ["processor.py", "operator.py", "tensor.py", "field.py", "logger.py"],
        "processing": ["coherence.py", "field.py", "quantum.py", "reality.py", "state.py", "tensor.py"],
        "processors": ["acceleration.py", "harmonics.py", "integration.py", "interface.py", 
                      "measurement.py", "precision.py", "synchronization.py", "validation.py"],
        "integration": ["emotion.py", "personality.py", "unified.py", "voice.py"],
        "optimization": ["coherence_optimizer.py", "field_optimizer.py", "optimization_processor.py", 
                        "reality_optimizer.py", "state_optimizer.py"],
        "orchestration": ["coherence_orchestrator.py", "field_orchestrator.py", "reality_orchestrator.py", 
                         "state_orchestrator.py", "unified_orchestrator.py"],
        "reality": ["coherence.py", "interface.py", "manager.py", "stabilizer.py"],
        "stabilization": ["coherence.py", "field.py", "reality.py", "stability.py", "state.py"],
        "validation": ["coherence.py", "field.py", "reality.py", "state.py"]
    }
    
    base_url = "https://raw.githubusercontent.com/XxDirectorxX/sebastian/main/backend/quantum_framework"
    base_dir = Path("R:/sebastian/backend/quantum_framework")
    
    # Create backup of any existing files
    if base_dir.exists():
        backup_dir = Path("R:/sebastian/backend/quantum_framework_backup")
        shutil.copytree(base_dir, backup_dir, dirs_exist_ok=True)
        print(f"Created backup at {backup_dir}")
    
    # Recover each directory
    for dir_name, files in directories.items():
        dir_path = base_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py for each directory
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            init_file.touch()
        
        # Recover each file
        for file_name in files:
            file_url = f"{base_url}/{dir_name}/{file_name}"
            file_path = dir_path / file_name
            
            try:
                # Try to get file from GitHub
                response = requests.get(file_url)
                if response.status_code == 200:
                    file_path.write_text(response.text)
                    print(f"Recovered: {dir_name}/{file_name}")
                else:
                    print(f"Could not recover {dir_name}/{file_name} from GitHub")
            except Exception as e:
                print(f"Error recovering {dir_name}/{file_name}: {e}")

if __name__ == "__main__":
    recover_framework_directories()