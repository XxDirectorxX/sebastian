from pathlib import Path
import cv2
import json
import time
import numpy as np
from scenedetect import detect, ContentDetector
import subprocess
import pysrt
from PIL import Image
import pytesseract

class VideoProcessor:
    def __init__(self):
        self.video_dir = Path(r"R:\sebastian\backend\quantum_framework\personality\training\data_collection\raw_data\video")
        self.results_dir = Path(r"R:\sebastian\backend\quantum_framework\personality\training\data_collection\analysis")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize scene detection thresholds
        self.scene_threshold = 30
        self.motion_threshold = 50

        self.personality_layers = {
            'primary': ['loyal', 'intelligent', 'polite'],
            'secondary': ['ruthless', 'sadistic', 'sarcastic'],
            'core_traits': ['perfectionist', 'manipulative', 'emotionally_detached']
        }
        
        self.trait_indicators = {
            'loyal': ['protecting Ciel', 'following orders', 'butler duties'],
            'intelligent': ['strategy', 'problem solving', 'knowledge display'],
            'polite': ['formal speech', 'proper etiquette', 'butler manners'],
            'ruthless': ['combat efficiency', 'demon form', 'threat elimination'],
            'sadistic': ['toying with enemies', 'dark humor', 'enjoying chaos'],
            'sarcastic': ['witty remarks', 'subtle mockery', 'veiled insults'],
            'perfectionist': ['precise movements', 'attention to detail', 'flawless execution'],
            'manipulative': ['subtle influence', 'psychological tactics', 'orchestrating events'],
            'emotionally_detached': ['calm demeanor', 'unfazed reactions', 'clinical responses']
        }

        # Add to video analysis structure
        self.analysis_categories = {
            'personality_traits': {
                'primary_traits': [],
                'secondary_traits': [],
                'core_traits': []
            }
        }

    def analyze_personality_traits(self, frame, dialogue, timestamp):
        for layer, traits in self.personality_layers.items():
            for trait in traits:
                indicators = self.trait_indicators[trait]
                if self.detect_trait_indicators(frame, dialogue, indicators):
                    self.analysis_categories['personality_traits'][layer].append({
                        'trait': trait,
                        'timestamp': timestamp,
                        'context': dialogue or 'visual cue'
                    })

    def detect_trait_indicators(self, frame, dialogue, indicators):
        # Visual analysis for trait indicators
        # Dialogue analysis for trait markers
        # Return True if indicators are detected
        pass        
    def process_videos(self):
        video_analysis = {
            'key_scenes': [],
            'dialogue_timestamps': [],
            'behavioral_moments': [],
            'combat_sequences': [],
            'butler_duties': [],
            'scene_metadata': {}
        }

        for video_file in self.video_dir.glob('*.mp4'):
            print(f"Processing: {video_file.name}")
            self.analyze_video(video_file, video_analysis)

        self.save_analysis(video_analysis)
        return video_analysis

    def analyze_video(self, video_path, analysis):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Scene detection
        scenes = self.detect_scenes(video_path)
        analysis['scene_metadata'][video_path.stem] = {
            'scenes': scenes,
            'fps': fps,
            'total_frames': total_frames
        }
        
        # Extract subtitles
        subtitles = self.extract_subtitles(video_path)
        dialogue_data = self.process_dialogue(subtitles)
        analysis['dialogue_timestamps'].extend(dialogue_data)
        
        # Process frame by frame
        prev_frame = None
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Motion detection
            if prev_frame is not None:
                motion = self.detect_motion(prev_frame, frame)
                if motion > self.motion_threshold:
                    self.analyze_action_sequence(frame, frame_count/fps, analysis)
            
            # Character detection
            if frame_count % 30 == 0:  # Check every 30 frames
                self.detect_character_presence(frame, frame_count/fps, analysis)
            
            # Butler duty detection
            self.detect_butler_duties(frame, frame_count/fps, analysis)
            
            prev_frame = frame
            frame_count += 1
            
        cap.release()

    def detect_scenes(self, video_path):
        scenes = detect(str(video_path), ContentDetector())
        return [(scene[0].get_frames(), scene[1].get_frames()) for scene in scenes]

    def extract_subtitles(self, video_path):
        srt_path = video_path.with_suffix('.srt')
        if srt_path.exists():
            return pysrt.open(str(srt_path))
        return []

    def process_dialogue(self, subtitles):
        dialogue_data = []
        for sub in subtitles:
            if "Sebastian" in sub.text:
                dialogue_data.append({
                    'timestamp': (sub.start.seconds, sub.end.seconds),
                    'text': sub.text,
                    'speaker': 'Sebastian'
                })
        return dialogue_data

    def detect_motion(self, prev_frame, curr_frame):
        diff = cv2.absdiff(prev_frame, curr_frame)
        return np.mean(diff)

    def analyze_action_sequence(self, frame, timestamp, analysis):
        # Convert frame to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect rapid movement patterns
        edges = cv2.Canny(gray, 100, 200)
        if np.mean(edges) > self.scene_threshold:
            analysis['combat_sequences'].append({
                'timestamp': timestamp,
                'intensity': np.mean(edges)
            })

    def detect_character_presence(self, frame, timestamp, analysis):
        # Convert frame to RGB for character detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use OCR to detect text that might indicate Sebastian's presence
        text = pytesseract.image_to_string(Image.fromarray(rgb_frame))
        if "Sebastian" in text:
            analysis['behavioral_moments'].append({
                'timestamp': timestamp,
                'type': 'character_presence',
                'details': text
            })

    def detect_butler_duties(self, frame, timestamp, analysis):
        # Convert frame to HSV for color-based detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for butler uniform (black)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])
        
        # Create mask for butler uniform
        mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # If significant black areas detected, might be butler duty scene
        if np.sum(mask) > frame.shape[0] * frame.shape[1] * 0.3:
            analysis['butler_duties'].append({
                'timestamp': timestamp,
                'type': 'uniform_detected'
            })

    def save_analysis(self, analysis):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Save JSON analysis
        json_file = self.results_dir / f'video_analysis_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
            
        # Save readable report
        report_file = self.results_dir / f'video_report_{timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== VIDEO ANALYSIS REPORT ===\n\n")
            
            f.write("KEY SCENES DETECTED:\n")
            for scene in analysis['key_scenes']:
                f.write(f"- Scene at {scene['timestamp']}\n")
                
            f.write("\nDIALOGUE TIMESTAMPS:\n")
            for dialogue in analysis['dialogue_timestamps']:
                f.write(f"- {dialogue['timestamp']}: {dialogue['text']}\n")
                
            f.write("\nCOMBAT SEQUENCES:\n")
            for combat in analysis['combat_sequences']:
                f.write(f"- Combat at {combat['timestamp']}, Intensity: {combat['intensity']}\n")
                
            f.write("\nBUTLER DUTIES:\n")
            for duty in analysis['butler_duties']:
                f.write(f"- Duty detected at {duty['timestamp']}\n")

if __name__ == "__main__":
    processor = VideoProcessor()
    processor.process_videos()
