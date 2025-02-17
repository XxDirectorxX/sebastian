import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional

class SceneAnalyzer:
    def __init__(self, scene_data: Union[str, dict], output_dir: Optional[Path] = None):
        """Initialize SceneAnalyzer with scene data and optional output directory."""
        self.scenes = json.loads(scene_data) if isinstance(scene_data, str) else scene_data
        self.df = pd.DataFrame(self.scenes)
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_key_events(self) -> List[Dict]:
        """Extract key events from the scene data with improved detection."""
        events = []
        previous_row = None
        
        for idx, row in self.df.iterrows():
            if previous_row is not None:
                self._check_state_changes(previous_row, row, events)
            previous_row = row
            
        return sorted(events, key=lambda x: x['timestamp'])
    
    def _check_state_changes(self, prev_row: pd.Series, curr_row: pd.Series, events: List[Dict]):
        """Helper method to check state changes between frames."""
        if curr_row['has_faces'] != prev_row['has_faces']:
            events.append({
                'timestamp': curr_row['timestamp'],
                'event': 'Face detected' if curr_row['has_faces'] else 'Face lost',
                'confidence': curr_row.get('face_confidence', None)
            })
            
        if curr_row['has_motion'] != prev_row['has_motion']:
            events.append({
                'timestamp': curr_row['timestamp'],
                'event': 'Motion started' if curr_row['has_motion'] else 'Motion stopped',
                'intensity': curr_row.get('motion_intensity', None)
            })

    def _json_serializer(self, obj):
        """Custom JSON serializer to handle numpy types"""
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        if isinstance(obj, (float, np.floating)):
            return float(obj)
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')
    
    def generate_statistics(self) -> Dict:
        """Generate comprehensive statistics with additional metrics."""
        total_frames = len(self.df)
        stats = {
            'total_duration': float(self.df['timestamp'].max()),
            'frame_count': int(total_frames),
            'fps': float(total_frames / self.df['timestamp'].max()),
            'face_detection': {
                'time_percentage': float((self.df['has_faces'].sum() / total_frames) * 100),
                'total_occurrences': int(self.df['has_faces'].sum()),
                'average_duration': float(self._calculate_average_duration('has_faces'))
            },
            'motion_detection': {
                'time_percentage': float((self.df['has_motion'].sum() / total_frames) * 100),
                'total_occurrences': int(self.df['has_motion'].sum()),
                'average_duration': float(self._calculate_average_duration('has_motion'))
            },
            'scene_types': {k: int(v) for k, v in self.df['scene_type'].value_counts().to_dict().items()}
        }
        return stats
    
    def _calculate_average_duration(self, column: str) -> float:
        """Calculate average duration of continuous detection periods."""
        durations = []
        start_time = None
        
        for idx, row in self.df.iterrows():
            if row[column] and start_time is None:
                start_time = row['timestamp']
            elif not row[column] and start_time is not None:
                durations.append(row['timestamp'] - start_time)
                start_time = None
            
        if start_time is not None:
            durations.append(self.df['timestamp'].iloc[-1] - start_time)
            
        return sum(durations) / len(durations) if durations else 0
    
    def plot_timeline(self, show_grid: bool = True):
        """Generate an enhanced timeline visualization."""
        plt.style.use('seaborn')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        
        # Face detection subplot
        ax1.plot(self.df['timestamp'], self.df['has_faces'], 'b-', label='Faces')
        ax1.set_ylabel('Face Detection')
        ax1.set_title('Scene Analysis Timeline')
        ax1.grid(show_grid)
        ax1.legend()
        
        # Motion detection subplot
        ax2.plot(self.df['timestamp'], self.df['has_motion'], 'r-', label='Motion')
        ax2.set_xlabel('Timestamp (s)')
        ax2.set_ylabel('Motion Detection')
        ax2.grid(show_grid)
        ax2.legend()
        
        plt.tight_layout()
        output_path = self.output_dir / 'scene_timeline.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_report(self) -> str:
        """Generate a detailed analysis report."""
        stats = self.generate_statistics()
        events = self.get_key_events()
        
        report = f"""
Scene Analysis Report
====================
Analysis Summary:
----------------
Total Duration: {stats['total_duration']:.2f} seconds
Total Frames: {stats['frame_count']}
Average FPS: {stats['fps']:.2f}

Detection Statistics:
-------------------
Face Detection:
- Active Time: {stats['face_detection']['time_percentage']:.1f}% of total duration
- Total Occurrences: {stats['face_detection']['total_occurrences']}
- Average Duration: {stats['face_detection']['average_duration']:.2f} seconds

Motion Detection:
- Active Time: {stats['motion_detection']['time_percentage']:.1f}% of total duration
- Total Occurrences: {stats['motion_detection']['total_occurrences']}
- Average Duration: {stats['motion_detection']['average_duration']:.2f} seconds

Scene Type Distribution:
----------------------"""
        
        for scene_type, count in stats['scene_types'].items():
            report += f"\n- {scene_type}: {count} frames"
            
        report += "\n\nKey Events Timeline:\n------------------"
        for event in events:
            details = []
            if 'confidence' in event and event['confidence'] is not None:
                details.append(f"confidence: {event['confidence']:.2f}")
            if 'intensity' in event and event['intensity'] is not None:
                details.append(f"intensity: {event['intensity']:.2f}")
                
            detail_str = f" ({', '.join(details)})" if details else ""
            report += f"\n- {event['event']}{detail_str} at {event['timestamp']:.3f}s"
            
        return report


    def save_results(self):
        """Save all analysis results to files."""
        # Save report
        report_path = self.output_dir / 'analysis_report.txt'
        report_path.write_text(self.generate_report())
        # Save statistics
        stats_path = self.output_dir / 'scene_stats.json'
        with stats_path.open('w') as f:
            json.dump(self.generate_statistics(), f, indent=4, default=self._json_serializer)
        
        # Generate and save timeline plot
        self.plot_timeline()

if __name__ == "__main__":
    # Load the JSON file correctly
    json_path = Path("R:/sebastian/backend/quantum_framework/personality/training/data_collection/raw_data/text/scripts")
    
    for json_file in json_path.glob("*_analysis.json"):
        with open(json_file, 'r') as f:
            scene_data = json.load(f)
        output_dir = Path('analysis_results') / json_file.stem
        analyzer = SceneAnalyzer(scene_data, output_dir=output_dir)
        analyzer.save_results()