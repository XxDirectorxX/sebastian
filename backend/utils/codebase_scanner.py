import os
import ast
from pathlib import Path
from typing import Dict, Set
from collections import defaultdict
from datetime import datetime
import gzip
import json
from itertools import islice


class CodebaseScanner:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.scan_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.scan_dir = Path("R:/sebastian/scans")
        self.scan_dir.mkdir(exist_ok=True)
        
        # Extension tracking
        self.extensions = defaultdict(set)
        self.file_counts = defaultdict(int)
        
        # Import tracking 
        self.imports = defaultdict(set)
        self.import_counts = defaultdict(int)

    def scan_imports(self, file_path: Path) -> None:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
                
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        self.imports[name.name].add(str(file_path))
                        self.import_counts[name.name] += 1
                elif isinstance(node, ast.ImportFrom):
                    module = node.module if node.module else ''
                    for name in node.names:
                        full_import = f"{module}.{name.name}" if module else name.name
                        self.imports[full_import].add(str(file_path))
                        self.import_counts[full_import] += 1
        except Exception as e:
            print(f"Error scanning imports in {file_path}: {e}")

    def scan_directory(self) -> None:
        """Scans entire directory for files and imports"""
        for path in self.root_dir.rglob('*'):
            if path.is_file():
                # Track extensions
                ext = path.suffix.lower()
                self.extensions[ext or 'NO_EXTENSION'].add(str(path))
                self.file_counts[ext or 'NO_EXTENSION'] += 1
                
                # Track imports in Python files
                if ext == '.py':
                    self.scan_imports(path)

    def generate_report(self):
        """Generates scan report in multiple txt files under 130kb"""
        base_path = self.scan_dir / f"codebase_scan_{self.scan_time}"
        
        # Split into separate text files
        self.write_summary(f"{base_path}_summary.txt")
        self.write_extensions(f"{base_path}_extensions.txt") 
        self.write_imports(f"{base_path}_imports.txt")
        
    def write_summary(self, path):
            with open(path, 'w') as f:
                f.write("Summary Report\n")
                # Write counts and statistics
            
    def write_extensions(self, path):
            with open(path, 'w') as f:
                for ext in self.extensions:
                    chunk = {ext: list(self.extensions[ext])}
                    f.write(f"{json.dumps(chunk)}\n")
                
    def write_imports(self, path): 
            with open(path, 'w') as f:
                for imp in self.imports:
                    chunk = {imp: list(self.imports[imp])}
                    f.write(f"{json.dumps(chunk)}\n")
# Run scanner
if __name__ == "__main__":
    scanner = CodebaseScanner("R:/sebastian")  # Changed from CodebaseScanner()
    scanner.scan_directory()
    scanner.generate_report()
