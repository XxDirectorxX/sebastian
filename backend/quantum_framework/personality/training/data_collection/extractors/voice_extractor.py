from pathlib import Path
import librosa
import pysrt
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from qiskit import QuantumCircuit, execute, Aer

class SebastianVoiceExtractor:
    def __init__(self):
        self.raw_dir = Path(r"R:\sebastian\backend\quantum_framework\personality\training\data_collection\raw_data")
        self.voice_output_dir = self.raw_dir / 'audio' / 'sebastian_voice'
        self.voice_output_dir.mkdir(exist_ok=True)
        
    def extract_voice_segments(self, video_file, subtitle_file):
        # Load audio from video
        audio = AudioSegment.from_file(video_file)
        
        # Load subtitles to identify Sebastian's lines
        subs = pysrt.open(subtitle_file)
        
        sebastian_voice_clips = []
        for sub in subs:
            if self.is_sebastian_speaking(sub.text):
                start_ms = sub.start.ordinal
                end_ms = sub.end.ordinal
                
                # Extract the audio segment
                voice_clip = audio[start_ms:end_ms]
                sebastian_voice_clips.append({
                    'audio': voice_clip,
                    'text': sub.text,
                    'timestamp': f"{start_ms}-{end_ms}"
                })
        
        return sebastian_voice_clips
    
    def is_sebastian_speaking(self, text):
        sebastian_patterns = [
            "yes, my lord",
            "one hell of a butler",
            "young master"
        ]
        return any(pattern in text.lower() for pattern in sebastian_patterns)
    
    def save_voice_clips(self, clips, video_name):
        for idx, clip in enumerate(clips):
            output_file = self.voice_output_dir / f"sebastian_voice_{video_name}_{idx}.wav"
            clip['audio'].export(output_file, format="wav")
