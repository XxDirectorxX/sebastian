from pathlib import Path
import speech_recognition as sr
import pysrt
import moviepy.editor as mp
import json

class SpeechExtractor:
    def __init__(self):
        self.video_dir = Path(r"R:\sebastian\backend\quantum_framework\personality\training\data_collection\raw_data\video")
        self.results_dir = Path(r"R:\sebastian\backend\quantum_framework\personality\training\data_collection\analysis")
        
        self.speech_patterns = {
            'greetings': ["good morning", "good evening", "welcome home"],
            'acknowledgments': ["yes, my lord", "as you wish", "it shall be done"],
            'mockery': ["my, my", "oh dear", "how troublesome"],
            'confidence': ["one hell of a butler", "what kind of butler would I be"],
            'reassurance': ["no need for concern", "according to plan"]
        }

    def extract_from_video(self, video_file):
        # Extract audio from video
        video = mp.VideoFileClip(str(video_file))
        audio = video.audio
        audio.write_audiofile("temp_audio.wav")
        
        # Process audio with speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile("temp_audio.wav") as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            
        # Process subtitles if available
        srt_file = video_file.with_suffix('.srt')
        if srt_file.exists():
            subtitles = pysrt.open(str(srt_file))
            
        return self.analyze_dialogue(text, subtitles)

    def analyze_dialogue(self, text, subtitles):
        dialogue_patterns = {
            'greetings': [],
            'acknowledgments': [],
            'mockery': [],
            'confidence': [],
            'reassurance': []
        }
        
        # Match patterns in speech and subtitles
        for category, patterns in self.speech_patterns.items():
            for pattern in patterns:
                if pattern in text.lower():
                    dialogue_patterns[category].append({
                        'text': text,
                        'source': 'audio'
                    })
                    
        # Process subtitle matches
        for sub in subtitles:
            for category, patterns in self.speech_patterns.items():
                for pattern in patterns:
                    if pattern in sub.text.lower():
                        dialogue_patterns[category].append({
                            'text': sub.text,
                            'timestamp': (sub.start.seconds, sub.end.seconds),
                            'source': 'subtitle'
                        })
                        
        return dialogue_patterns

    def process_all_videos(self):
        all_patterns = {}
        for video_file in self.video_dir.glob('*.mp4'):
            print(f"Processing {video_file.name}")
            patterns = self.extract_from_video(video_file)
            all_patterns[video_file.stem] = patterns
            
        self.save_results(all_patterns)

    def save_results(self, patterns):
        output_file = self.results_dir / 'speech_patterns.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, indent=2)

if __name__ == "__main__":
    extractor = SpeechExtractor()
    extractor.process_all_videos()
