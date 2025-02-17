import os
import cv2
import ffmpeg
import librosa
import pysrt
import json
import datetime
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.text import Text
from typing import Dict, List, Optional
from multiprocessing import Manager, Pool, Queue


class VideoExtractor:
    def __init__(self, video_files: Optional[List[Path]] = None):
        self.base_dir = Path(r"R:\sebastian\backend\quantum_framework\personality\training\data_collection")
        self.video_dir = self.base_dir / 'raw_data' / 'video'
        self.script_dir = self.base_dir / 'raw_data' / 'text' / 'scripts'
        self.script_dir.mkdir(parents=True, exist_ok=True)
        self.video_files = video_files if video_files else []
        
        if not self.video_files:  # Load video files from the directory if not passed
            for ext in ['*.mkv', '*.mp4']:
                self.video_files.extend(list(self.video_dir.rglob(ext)))

        self.cascade_path = r"C:\Users\Iam\anaconda3\envs\quantum-sys\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
        
    def initialize_face_cascade(self):
        return cv2.CascadeClassifier(self.cascade_path)  # Initialize within the worker function

    def identify_speaker(self, text: str) -> str:
        sebastian_patterns = [
            "yes, my lord",
            "indeed",
            "one hell of a butler",
            "young master"
        ]
        
        ciel_patterns = [
            "this is an order",
            "sebastian!",
            "phantomhive"
        ]
        
        text = text.lower()
        if any(pattern in text for pattern in sebastian_patterns):
            return "Sebastian"
        elif any(pattern in text for pattern in ciel_patterns):
            return "Ciel"
        return "Unknown"

    def extract_subtitles(self, video_file: Path) -> Optional[Path]:
        output_srt = video_file.with_suffix('.srt')
        if not output_srt.exists():
            try:
                stream = ffmpeg.input(str(video_file))
                stream = ffmpeg.output(stream, str(output_srt), map="0:s:0", c="copy")
                ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            except ffmpeg.Error:
                print(f"Could not extract subtitles from {video_file.name}")
                return None
        return output_srt

    def process_subtitles(self, srt_file: Path) -> List[Dict]:
        if not srt_file.exists():
            return []
        
        subs = pysrt.open(str(srt_file))
        dialogue_data = []
        
        for sub in subs:
            speaker = self.identify_speaker(sub.text)
            dialogue_data.append({
                'timestamp': sub.start.seconds,
                'speaker': speaker,
                'dialogue': sub.text
            })
            
        return dialogue_data

    def extract_audio(self, video_file: Path) -> Path:
        output_wav = video_file.with_suffix('.wav')
        if not output_wav.exists():
            stream = ffmpeg.input(str(video_file))
            stream = ffmpeg.output(stream, str(output_wav), acodec='pcm_s16le', ac=1, ar=16000)
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        return output_wav

    def isolate_character_voice(self, audio_file: Path, subtitle_data: List[Dict]) -> Dict[str, List[Dict]]:
        audio, sr = librosa.load(str(audio_file))
        character_voices = {'Sebastian': [], 'Ciel': [], 'Unknown': []}
        
        for entry in subtitle_data:
            start_sample = int(entry['timestamp'] * sr)
            end_sample = int((entry['timestamp'] + entry.get('duration', 0)) * sr)
            voice_segment = audio[start_sample:end_sample]
            
            character_voices[entry['speaker']].append({
                'audio': voice_segment,
                'text': entry['dialogue'],
                'timestamp': entry['timestamp']
            })
            
        return character_voices

    def get_frame_count(self, video_file: Path) -> int:
        cap = cv2.VideoCapture(str(video_file))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count

    def detect_faces(self, frame, face_cascade):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        return len(faces) > 0

    def detect_motion(self, frame, prev_frame):
        if prev_frame is None:
            return False
        
        diff = cv2.absdiff(frame, prev_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        significant_motion = False
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                significant_motion = True
                break
                
        return significant_motion    

    def analyze_scene(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_brightness = np.mean(hsv[:,:,2])
        
        if avg_brightness < 50:
            return "dark_scene"
        elif avg_brightness < 150:
            return "normal_scene"
        else:
            return "bright_scene"

    def save_analysis(self, video_name, frame_data):
        output_file = self.script_dir / f"{video_name}_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(frame_data, f)

    def generate_script(self, subtitle_data: List[Dict]) -> List[Dict]:
        script = []
        for entry in subtitle_data:
            script.append({
                'timestamp': entry['timestamp'],
                'speaker': entry['speaker'],
                'dialogue': entry['dialogue']
            })
        return script

    def analyze_video(self, video_file):
        return [f"frame_{i}" for i in range(100)]  # Example dummy frame data

    def process_single_video(self, video_file, queue):
        face_cascade = self.initialize_face_cascade()

        video_name = video_file.name
        frame_data = self.analyze_video(video_file)
        queue.put((video_name, frame_data))

        cap = cv2.VideoCapture(str(video_file))
        prev_frame = None
        frame_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 360))

            scene_data = {
                'timestamp': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                'has_faces': self.detect_faces(frame, face_cascade),
                'has_motion': self.detect_motion(frame, prev_frame),
                'scene_type': self.analyze_scene(frame)
            }

            frame_data.append(scene_data)
            prev_frame = frame.copy()

        cap.release()
        self.save_analysis(video_file.stem, frame_data)

        srt_file = self.extract_subtitles(video_file)
        if srt_file:
            subtitle_data = self.process_subtitles(srt_file)
            script_data = self.generate_script(subtitle_data)
            self.save_analysis(video_name, script_data)
        
        queue.put(f"Finished processing {video_name}")

    def process_videos(self):
        with Manager() as manager:  # Use Manager for shared memory
            queue = manager.Queue()  # Create a Queue
            with Pool() as pool:
                pool.starmap(self.process_single_video, [(video_file, queue) for video_file in self.video_files])

            while not queue.empty():
                result = queue.get()
                if isinstance(result, tuple):
                    video_name, frame_data = result
                    print(f"Processed {video_name} with {len(frame_data)} frames")
                else:
                    print(result)

if __name__ == "__main__":
    video_extractor = VideoExtractor()
    video_extractor.process_videos()