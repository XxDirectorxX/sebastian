import os
from pathlib import Path
from typing import Dict, Set
from collections import defaultdict
from datetime import datetime

class ExtensionScanner:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.extensions = defaultdict(set)
        self.file_counts = defaultdict(int)
        self.scan_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.scan_dir = Path("R:/sebastian/scans")
        self.scan_dir.mkdir(exist_ok=True)
        
    def scan_directory(self) -> Dict[str, int]:
        """Scans directory recursively and catalogs all file extensions"""
        for path in self.root_dir.rglob('*'):
            if path.is_file():
                ext = path.suffix.lower()
                if ext:
                    self.extensions[ext].add(str(path))
                    self.file_counts[ext] += 1
        return dict(self.file_counts)
    
    def generate_report(self):
        """Generates detailed scan report"""
        report_path = self.scan_dir / f"extension_scan_{self.scan_time}.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Sebastian Codebase Extension Scan\n")
            f.write(f"Scan Time: {self.scan_time}\n")
            f.write("-" * 50 + "\n\n")
            
            f.write("Extension Summary:\n")
            f.write("-" * 20 + "\n")
            for ext, count in sorted(self.file_counts.items()):
                f.write(f"{ext:8} : {count:5} files\n")
            
            f.write("\nDetailed File Listings:\n")
            f.write("-" * 20 + "\n")
            for ext in sorted(self.extensions.keys()):
                f.write(f"\n{ext} files:\n")
                for file_path in sorted(self.extensions[ext]):
                    f.write(f"  {file_path}\n")
                    
        print(f"\nScan report generated: {report_path}")

    def print_summary(self):
        """Prints summary to terminal"""
        print("\nSebastian Codebase Extension Scan")
        print(f"Scan Time: {self.scan_time}")
        print("-" * 50)
        
        for ext, count in sorted(self.file_counts.items()):
            print(f"{ext:8} : {count:5} files")

# Run scanner
if __name__ == "__main__":
    scanner = ExtensionScanner("R:/sebastian")
    scanner.scan_directory()
    scanner.print_summary()
    scanner.generate_report()
