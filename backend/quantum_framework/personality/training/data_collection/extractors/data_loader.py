from pathlib import Path

class DataLoader:
    def __init__(self):
        self.analysis_dir = Path(r"R:\sebastian\backend\quantum_framework\personality\training\data_collection\analysis")
        
    def load_latest_analysis(self):
        # Get most recent analysis files
        analysis_files = {
            'character_dialogue': self.get_latest_file('character_dialogue_*.json'),
            'video_analysis': self.get_latest_file('video_analysis_*.json'),
            'complete_analysis': self.get_latest_file('complete_analysis_*.json')
        }
        return analysis_files

    def get_latest_file(self, pattern):
        files = list(self.analysis_dir.glob(pattern))
        return max(files, key=lambda x: x.stat().st_mtime) if files else None
