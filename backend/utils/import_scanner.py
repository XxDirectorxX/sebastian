import os
import ast
from pathlib import Path
from typing import Dict, Set
from collections import defaultdict
from datetime import datetime

class ImportScanner:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.imports = defaultdict(set)
        self.import_counts = defaultdict(int)
        self.scan_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.scan_dir = Path("R:/sebastian/scans")
        self.scan_dir.mkdir(exist_ok=True)

    def scan_file(self, file_path: Path) -> None:
        """Extracts imports from a Python file"""
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
            print(f"Error scanning {file_path}: {e}")

    def scan_directory(self) -> Dict[str, int]:
        """Scans directory recursively for Python imports"""
        for path in self.root_dir.rglob('*.py'):
            self.scan_file(path)
        return dict(self.import_counts)

    def generate_report(self):
        """Generates detailed import scan report"""
        report_path = self.scan_dir / f"import_scan_{self.scan_time}.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Sebastian Codebase Import Scan\n")
            f.write(f"Scan Time: {self.scan_time}\n")
            f.write("-" * 50 + "\n\n")
            
            f.write("Import Summary:\n")
            f.write("-" * 20 + "\n")
            for imp, count in sorted(self.import_counts.items()):
                f.write(f"{imp:40} : {count:5} occurrences\n")
            
            f.write("\nDetailed Import Locations:\n")
            f.write("-" * 20 + "\n")
            for imp in sorted(self.imports.keys()):
                f.write(f"\n{imp}:\n")
                for file_path in sorted(self.imports[imp]):
                    f.write(f"  {file_path}\n")

        print(f"\nImport scan report generated: {report_path}")

    def print_summary(self):
        """Prints import summary to terminal"""
        print("\nSebastian Codebase Import Scan")
        print(f"Scan Time: {self.scan_time}")
        print("-" * 50)
        
        for imp, count in sorted(self.import_counts.items()):
            print(f"{imp:40} : {count:5} occurrences")

# Run scanner
if __name__ == "__main__":
    scanner = ImportScanner("R:/sebastian")
    scanner.scan_directory()
    scanner.print_summary()
    scanner.generate_report()
