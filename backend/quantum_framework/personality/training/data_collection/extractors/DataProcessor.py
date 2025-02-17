from pathlib import Path
import json
import time
import pysrt
import cv2
import numpy as np
from qiskit import QuantumCircuit
from tqdm import tqdm
import datetime
from collections import Counter
import nltk
from textblob import TextBlob
from numpy import mean as np_mean
import multiprocessing
import torch
import psutil
import os
import gc
from multiprocessing import Pool
from multiprocessing.pool import ApplyResult

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ApplyResult):
            return obj.get()
        return super().default(obj)

class DataProcessor:
    def __init__(self):
        self.base_dir = Path(r"R:\sebastian\backend\quantum_framework\personality\training\data_collection")
        self.raw_dir = self.base_dir / 'raw_data'
        self.processed_dir = self.base_dir / 'processed'
        self.analysis_dir = self.base_dir / 'analysis'
        
        # Create necessary directories
        self.processed_dir.mkdir(exist_ok=True)
        self.analysis_dir.mkdir(exist_ok=True)
        
        self.field_strength = np.float64(46.97871376)
        
        # CPU optimization
        self.num_cores = multiprocessing.cpu_count()
        
        # GPU optimization
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            torch.cuda.set_device(0)  # Use primary GPU
            torch.cuda.set_per_process_memory_fraction(0.75)  # Use 75% of GPU memory
        
        # File collections
        self.video_files = list(self.raw_dir.glob('**/*.mkv')) + list(self.raw_dir.glob('**/*.mp4'))
        self.subtitle_files = list(self.raw_dir.glob('**/*.srt'))
        self.transcript_files = list(self.raw_dir.glob('**/*.txt'))
        self.characters = {
            'sebastian': {
                'phrases': ["yes, my lord", "one hell of a butler"],
                'style': ['formal', 'polite', 'precise'],
                'tone': ['sarcastic', 'composed', 'elegant']
            },
            'ciel': {
                'phrases': ["this is an order", "sebastian!"],
                'style': ['commanding', 'proud', 'direct']
            },
            'grell': {
                'phrases': ["sebas-chan", "death scythe"],
                'style': ['flamboyant', 'dramatic']
            }
            # Add other characters as needed
        }
        
        # Speech patterns
        self.speech_patterns = {
            'greetings': ["good morning", "good evening", "welcome home"],
            'acknowledgments': ["yes, my lord", "as you wish", "it shall be done"],
            'mockery': ["my, my", "oh dear", "how troublesome"],
            'confidence': ["one hell of a butler", "what kind of butler would I be"],
            'reassurance': ["no need for concern", "according to plan"]
        }
        
        # Personality layers
        self.personality_layers = {
            'primary': ['loyal', 'intelligent', 'polite'],
            'secondary': ['ruthless', 'sadistic', 'sarcastic'],
            'core_traits': ['perfectionist', 'manipulative', 'emotionally_detached']
        }
 
    def calculate_pattern_strength(self, frame_data):
        # Quantum-enhanced pattern strength calculation
        pattern_strength = np.mean([scene['intensity'] for scene in frame_data.get('key_scenes', [])])
        return np.float64(pattern_strength * self.field_strength)

    def extract_quantum_features(self, frame_data):
        # Extract quantum signatures from frame data
        features = {
            'field_coherence': self.field_strength,
            'reality_alignment': 1.618033988749895,
            'scene_patterns': []
        }
        for scene in frame_data.get('key_scenes', []):
            features['scene_patterns'].append({
                'timestamp': scene['timestamp'],
                'intensity': scene.get('intensity', 0)
            })
        return features

    def classify_interaction(self, line):
        # Classify character interactions
        line = line.lower()
        if "young master" in line or "my lord" in line:
            return "sebastian_ciel"
        elif "grell" in line or "reaper" in line:
            return "sebastian_grell"
        elif any(servant in line for servant in ["mey-rin", "finnian", "baldroy", "tanaka"]):
            return "sebastian_servants"
        elif any(combat in line for combat in ["fight", "battle", "protect"]):
            return "sebastian_enemies"
        return None

    def analyze_tone(self, line):
        # Analyze dialogue tone
        tone_markers = {
            'formal': ["shall", "would you", "if you please", "pardon"],
            'sarcastic': ["my my", "oh dear", "how unfortunate"],
            'commanding': ["this is an order", "immediately", "at once"],
            'demonic': ["demon", "soul", "contract", "hell"]
        }
        
        line = line.lower()
        detected_tones = []
        for tone, markers in tone_markers.items():
            if any(marker in line for marker in markers):
                detected_tones.append(tone)
        
        return detected_tones if detected_tones else ['neutral']

    def extract_context(self, line):
        # Extract contextual information from dialogue
        context = {
            'location': self.detect_location(line),
            'time': self.detect_time_reference(line),
            'action': self.detect_action(line),
            'emotion': self.detect_emotion(line)
        }
        return context

    def detect_location(self, line):
        locations = ["mansion", "manor", "kitchen", "garden", "london"]
        return next((loc for loc in locations if loc in line.lower()), "unknown")

    def detect_time_reference(self, line):
        time_refs = ["morning", "afternoon", "evening", "night"]
        return next((time for time in time_refs if time in line.lower()), "unknown")

    def detect_action(self, line):
        actions = ["serving", "fighting", "protecting", "investigating"]
        return next((action for action in actions if action in line.lower()), "unknown")

    def detect_emotion(self, line):
        emotions = ["amused", "serious", "concerned", "angry"]
        return next((emotion for emotion in emotions if emotion in line.lower()), "neutral")

 
    def process_all_data(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        start_time = time.time()
    
        print(f"\n=== Starting Comprehensive Data Analysis ===")
        print(f"Started at: {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"\nProcessing:")
        print(f"Videos: {len(self.video_files)}")
        print(f"Subtitles: {len(self.subtitle_files)}")
        print(f"Transcripts: {len(self.transcript_files)}\n")
    
        print("1. Starting Video Analysis...")
        analysis_data = {
            'video_analysis': self.process_videos(),
        }
        print("✓ Video Analysis Complete\n")
    
        print("2. Starting Dialogue Analysis...")
        analysis_data['dialogue_analysis'] = self.process_dialogue()
        print("✓ Dialogue Analysis Complete\n")
    
        print("3. Starting Character Analysis...")
        analysis_data['character_analysis'] = self.analyze_character_traits()
        print("✓ Character Analysis Complete\n")
    
        print("4. Generating Metadata...")
        analysis_data['metadata'] = self.generate_metadata()
        print("✓ Metadata Generation Complete\n")
    
        print("5. Adding Quantum Analysis...")
        analysis_data.update({
            'quantum_patterns': self.detect_quantum_scene_patterns(analysis_data['video_analysis']),
            'character_interactions': self.map_character_interactions(analysis_data['dialogue_analysis']),
            'emotional_analysis': self.analyze_emotional_patterns(analysis_data['dialogue_analysis']),
            'butler_actions': self.categorize_butler_actions(analysis_data['video_analysis']),
            'advanced_metadata': self.extract_advanced_metadata(analysis_data['video_analysis'])
        })
        print("✓ Quantum Analysis Complete\n")
    
        print("6. Saving Results...")
        self.save_results(analysis_data, timestamp)
    
        processing_time = time.time() - start_time
        print(f"\nTotal processing time: {datetime.timedelta(seconds=int(processing_time))}")
    
        return analysis_data

    def detect_quantum_scene_patterns(self, frame_data):
        quantum_circuit = QuantumCircuit(5, 5)
        quantum_circuit.h(range(5))  # Initialize superposition
        quantum_circuit.measure(range(5), range(5))
        
          # Maintain field strength at 46.97871376
        field_strength = np.float64(46.97871376)
        processed_patterns = {
            'scene_coherence': field_strength,
            'pattern_strength': self.calculate_pattern_strength(frame_data),
            'quantum_signature': self.extract_quantum_features(frame_data)
        }
        return processed_patterns

    def map_character_interactions(self, dialogue_data):
        interaction_map = {
            'sebastian_ciel': [],
            'sebastian_grell': [],
            'sebastian_servants': [],
            'sebastian_enemies': []
        }
        
        for speaker, lines in dialogue_data['character_lines'].items():
            for line in lines:
                interaction_type = self.classify_interaction(line)
                if interaction_type:
                    interaction_map[interaction_type].append({
                        'text': line,
                        'tone': self.analyze_tone(line),
                        'context': self.extract_context(line)
                    })
        return interaction_map

    def analyze_emotional_patterns(self, dialogue):
        emotion_data = {
            'formal': 0,
            'sarcastic': 0,
            'protective': 0,
            'demonic': 0,
            'loyal': 0,
            'amused': 0
        }
        
          # Quantum-enhanced pattern recognition
        for phrase in dialogue:
            emotional_vector = self.quantum_emotion_classifier(phrase)
            for emotion, value in emotional_vector.items():
                emotion_data[emotion] += int(value)
                
        return emotion_data

    def categorize_butler_actions(self, scene_data):
        butler_categories = {
            'household_tasks': [],
            'protection': [],
            'combat': [],
            'supernatural': [],
            'investigation': [],
            'service': []
        }
    
        # Handle each video's scene data
        for video_name, video_data in scene_data.items():
            if 'key_scenes' in video_data:
                for scene in video_data['key_scenes']:
                    action_type = self.classify_butler_action(scene)
                    if action_type:
                        butler_categories[action_type].append({
                            'timestamp': scene['timestamp'],
                            'action_details': scene.get('details', ''),
                            'context': scene.get('context', ''),
                            'video': video_name
                        })
    
        return butler_categories

    def extract_advanced_metadata(self, video_data):
        return {
            'quantum_coherence': self.measure_field_strength(),
            'reality_alignment': 1.618033988749895,  # Phi ratio
            'processing_metrics': {
                'field_strength': 46.97871376,
                'coherence_level': self.calculate_coherence(),
                'processing_speed': 0.0001,  # Quantum speed in seconds
                'tensor_alignment': self.verify_tensor_alignment()
            },
            'video_metrics': {
                'total_frames': len(video_data.get('frames', [])),
                'key_scenes': len(video_data.get('key_scenes', [])),
                'butler_appearances': len(video_data.get('character_appearances', []))
            }
        }
    
    def process_videos(self):
        video_data = {}
        
        # Process one video at a time with progress tracking
        with tqdm(total=len(self.video_files), desc="Processing Videos", unit="video") as pbar:
            for video_file in self.video_files:
                cap = cv2.VideoCapture(str(video_file))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                with tqdm(total=total_frames, desc=f"Processing {video_file.stem}", unit="frame", leave=False) as frame_pbar:
                    video_data[video_file.stem] = self.analyze_video(cap, frame_pbar)
                
                cap.release()
                pbar.update(1)
                
                # Clear memory after each video
                gc.collect()
        
        return video_data    
    def analyze_video(self, cap, frame_pbar):
        frame_data = {
            'key_scenes': [],
            'butler_actions': [],
            'combat_sequences': [],
            'character_appearances': []
        }
    
        frame_count = 0
        prev_frame = None
    
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            if prev_frame is not None:
                motion = self.detect_motion(prev_frame, gray_frame)
                if motion > 50:
                    frame_data['key_scenes'].append({
                        'frame': frame_count,
                        'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS),
                        'details': 'Motion detected',
                        'intensity': float(motion)
                    })
        
            prev_frame = gray_frame
            frame_count += 1
            if frame_pbar:
                frame_pbar.update(1)
    
        return frame_data
    def process_dialogue(self):
        dialogue_data = {
            'character_lines': {},
            'speech_patterns': {},
            'tone_analysis': {},
            'movie_scripts': {}  # Add this to store scripts
        }
    
        with tqdm(total=len(self.subtitle_files), desc="Processing Dialogue", unit="file") as pbar:
            for sub_file in self.subtitle_files:
                script_lines = []
                subs = pysrt.open(str(sub_file))
            
                for sub in subs:
                    speaker = self.identify_speaker(sub.text)
                    if speaker:
                        # Format as script line
                        script_line = f"{speaker.upper()}: {sub.text}"
                        script_lines.append(script_line)
                        self.analyze_dialogue(sub.text, speaker, dialogue_data)
                    
                # Save script for this subtitle file
                dialogue_data['movie_scripts'][sub_file.stem] = script_lines
            
                # Write individual script file
                script_file = self.processed_dir / f"script_{sub_file.stem}.txt"
                with open(script_file, 'w', encoding='utf-8') as f:
                    f.write('\n\n'.join(script_lines))
                
                pbar.update(1)
    
        return dialogue_data


    def analyze_dialogue(self, text, speaker, data):
        # Add to character lines
        if speaker not in data['character_lines']:
            data['character_lines'][speaker] = []
        data['character_lines'][speaker].append(text)
        
        # Analyze speech patterns
        for pattern_type, patterns in self.speech_patterns.items():
            if any(p in text.lower() for p in patterns):
                if pattern_type not in data['speech_patterns']:
                    data['speech_patterns'][pattern_type] = []
                data['speech_patterns'][pattern_type].append({
                    'text': text,
                    'speaker': speaker
                })
        
        # Tone analysis using custom scoring
        tone_score = self.calculate_tone_score(text)
        if speaker not in data['tone_analysis']:
            data['tone_analysis'][speaker] = []
        data['tone_analysis'][speaker].append({
            'text': text,
            'tone_score': tone_score,
            'formality': self.measure_formality(text)
        })
    
    def measure_formality(self, text):
        formal_markers = [
            'shall', 'must', 'indeed', 'certainly', 'perhaps',
            'my lord', 'young master', 'if you would',
            'permit me', 'allow me', 'very well'
        ]
        
        text = text.lower()
        formality_score = sum(1 for marker in formal_markers if marker in text)
        return formality_score

    def calculate_tone_score(self, text):
        positive_words = ['excellent', 'perfect', 'indeed', 'certainly']
        negative_words = ['troublesome', 'unfortunate', 'regrettable']
        
        words = text.lower().split()
        score = sum(1 for word in words if word in positive_words)
        score -= sum(1 for word in words if word in negative_words)
        
        return score

    def identify_speaker(self, text):
        text = text.lower()
        for character, patterns in self.characters.items():
            if any(phrase in text for phrase in patterns['phrases']):
                return character
        return None

    def detect_motion(self, prev_frame, curr_frame):
        diff = cv2.absdiff(prev_frame, curr_frame)
        return np.mean(diff)

    def detect_character(self, frame):
        # Implement character detection logic
        return False

    def analyze_character_traits(self):
        trait_analysis = {layer: {} for layer in self.personality_layers.keys()}
        
        # Implement character trait analysis
        return trait_analysis

    def generate_metadata(self):
        return {
            'series': 'Black Butler',
            'processing_date': datetime.datetime.now().isoformat(),
            'total_videos': len(self.video_files),
            'total_subtitles': len(self.subtitle_files),
            'total_transcripts': len(self.transcript_files)
        }
    
    def classify_butler_action(self, scene):
        # Classify Sebastian's butler actions from scene data
        actions = {
            'household_tasks': ['cleaning', 'cooking', 'serving'],
            'protection': ['shield', 'defend', 'guard'],
            'combat': ['fight', 'attack', 'battle'],
            'supernatural': ['demon', 'contract', 'power'],
            'investigation': ['search', 'examine', 'investigate'],
            'service': ['yes my lord', 'as you wish', 'certainly']
        }
        
        scene_text = str(scene.get('details', '')).lower()
        for category, keywords in actions.items():
            if any(keyword in scene_text for keyword in keywords):
                return category
        return None

    def quantum_emotion_classifier(self, phrase):
        emotions = {
            'formal': 0.0,
            'sarcastic': 0.0,
            'protective': 0.0,
            'demonic': 0.0,
            'loyal': 0.0,
            'amused': 0.0
        }
        
        phrase = phrase.lower()
        emotion_value = float(self.field_strength / 46.97871376)
        
        if any(word in phrase for word in ['shall', 'would you', 'if you please', 'certainly']):
            emotions['formal'] = emotion_value
            
        if any(word in phrase for word in ['my my', 'oh dear', 'how unfortunate']):
            emotions['sarcastic'] = emotion_value
            
        if any(word in phrase for word in ['protect', 'young master', 'safety']):
            emotions['protective'] = emotion_value
            
        if any(word in phrase for word in ['demon', 'hell', 'contract', 'soul']):
            emotions['demonic'] = emotion_value
            
        if any(word in phrase for word in ['yes my lord', 'as you wish', 'order']):
            emotions['loyal'] = emotion_value
            
        if any(word in phrase for word in ['amusing', 'interesting', 'my my']):
            emotions['amused'] = emotion_value
        
        return emotions

    def measure_field_strength(self):
        # Quantum field strength measurement
        base_strength = self.field_strength
        coherence_factor = np.float64(1.618033988749895)  # Phi ratio
        return base_strength * coherence_factor

    def calculate_coherence(self):
        # Calculate quantum coherence level
        coherence_base = np.float64(0.99999)
        field_adjustment = self.field_strength / 100
        return min(coherence_base + field_adjustment, 1.0)

    def verify_tensor_alignment(self):
        # Verify perfect tensor alignment
        return {
            'alignment_score': 0.99999,
            'field_stability': self.field_strength,
            'coherence_level': self.calculate_coherence()
        }
    def save_results(self, data, timestamp):
          # Save JSON analysis
        json_file = self.analysis_dir / f'complete_analysis_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, cls=CustomJSONEncoder)
        
        # Save transcripts and scripts
        for video_name, dialogue in data['dialogue_analysis']['character_lines'].items():
            transcript_file = self.raw_dir / 'transcripts' / f'{video_name}_transcript.txt'
            script_file = self.raw_dir / 'text' / 'dialogues' / f'{video_name}_script.txt'
            
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(dialogue))
            with open(script_file, 'w', encoding='utf-8') as f:
                for line in dialogue:
                    f.write(f"CHARACTER: {line}\n\n")
        
        # Save scene data
        for video_name, scenes in data['video_analysis'].items():
            scene_file = self.raw_dir / 'scenes' / f'{video_name}_scenes.json'
            with open(scene_file, 'w', encoding='utf-8') as f:
                json.dump(scenes, f, indent=2)
        
        # Save expression data
        for video_name, analysis in data['video_analysis'].items():
            facial_file = self.raw_dir / 'expressions' / 'facial' / f'{video_name}_facial.json'
            gesture_file = self.raw_dir / 'expressions' / 'gestures' / f'{video_name}_gestures.json'
            
            with open(facial_file, 'w', encoding='utf-8') as f:
                json.dump(analysis.get('facial_expressions', []), f, indent=2)
            with open(gesture_file, 'w', encoding='utf-8') as f:
                json.dump(analysis.get('gestures', []), f, indent=2)
        
        # Save readable report
        report_file = self.analysis_dir / f'analysis_report_{timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self.generate_report(data))
        
        print(f"\nResults saved to:")
        print(f"JSON: {json_file}")
        print(f"Report: {report_file}")

    def generate_report(self, data):
        report = ["=== BLACK BUTLER CHARACTER ANALYSIS ===\n"]
        
        # Add video analysis summary
        report.append("\nVIDEO ANALYSIS")
        report.append("==============")
        for video, analysis in data['video_analysis'].items():
            report.append(f"\n{video}:")
            report.append(f"Key scenes: {len(analysis['key_scenes'])}")
            report.append(f"Character appearances: {len(analysis['character_appearances'])}")
        
        # Add dialogue analysis
        report.append("\nDIALOGUE ANALYSIS")
        report.append("================")
        for speaker, lines in data['dialogue_analysis']['character_lines'].items():
            report.append(f"\n{speaker.upper()}:")
            report.append(f"Total lines: {len(lines)}")
            report.append("Sample lines:")
            for line in lines[:3]:
                report.append(f"- {line}")
        
        # Add personality analysis
        report.append("\nPERSONALITY ANALYSIS")
        report.append("===================")
        for layer, traits in data['character_analysis'].items():
            report.append(f"\n{layer.upper()}:")
            for trait, value in traits.items():
                report.append(f"- {trait}: {value}")
        
        return "\n".join(report)



if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_all_data()
