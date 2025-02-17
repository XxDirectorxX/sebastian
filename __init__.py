import sys
import os
from pathlib import Path
# This makes the directory a Python package
from .import_manager import *

# Add root directory to Python path
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

# Import and expose the import manager
from import_manager import *