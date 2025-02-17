from pathlib import Path
import json
import time
import pysrt

class DialogueIdentifier:
    def __init__(self):
        self.characters = {
            'sebastian': {
                'phrases': [
                    'yes, my lord',
                    'young master',
                    'one hell of a butler'
                ],
                'style': [
                    'shall i',
                    'permit me to',
                    'if you would'
                ]
            },
            'ciel': {
                'phrases': [
                    'this is an order',
                    'sebastian!',
                    'i am the head'
                ],
                'style': [
                    'phantomhive',
                    'queen\'s watchdog',
                    'contract'
                ]
            },
            'grell': {
                'phrases': [
                    'sebas-chan',
                    'death scythe',
                    'reaper'
                ],
                'style': [
                    'darling',
                    'red',
                    'gorgeous'
                ]
            },
            'mey-rin': {
                'phrases': [
                    'yes sir',
                    'mr sebastian',
                    'oh my'
                ],
                'style': [
                    'clumsy',
                    'glasses',
                    'yes i will'
                ]
            },
            'finnian': {
                'phrases': [
                    'mr sebastian',
                    'garden',
                    'strength'
                ],
                'style': [
                    'finny',
                    'sorry',
                    'plants'
                ]
            },
            'baldroy': {
                'phrases': [
                    'kitchen',
                    'cooking',
                    'explosion'
                ],
                'style': [
                    'bard',
                    'chef',
                    'military'
                ]
            },
            'tanaka': {
                'phrases': [
                    'ho ho ho',
                    'former butler',
                    'tea'
                ]
            },
            'madame_red': {
                'phrases': [
                    'sister\'s son',
                    'red',
                    'angelina'
                ]
            },
            'lau': {
                'phrases': [
                    'earl',
                    'interesting',
                    'chinese'
                ],
                'style': [
                    'opium',
                    'ran mao',
                    'trade'
                ]
            }
        }

    def process_transcript_directory(self):
        transcript_dir = Path(r"R:\sebastian\backend\quantum_framework\personality\training\data_collection\raw_data\transcripts")
        results = {char: [] for char in self.characters.keys()}
        
        for transcript in transcript_dir.glob('*.srt'):
            subs = pysrt.open(str(transcript))
            for sub in subs:
                speaker = self.identify_speaker(sub.text)
                if speaker != 'unknown':
                    results[speaker].append({
                        'text': sub.text,
                        'episode': transcript.stem,
                        'timestamp': (sub.start.seconds, sub.end.seconds)
                    })
        
        return results

    def identify_speaker(self, line):
        line = line.lower()
        for char, patterns in self.characters.items():
            # Check character's signature phrases
            if any(phrase.lower() in line for phrase in patterns['phrases']):
                return char
            # Check speaking style if available
            if 'style' in patterns:
                if any(style.lower() in line for style in patterns['style']):
                    return char
        return 'unknown'

    def save_dialogue_analysis(self, results):
        output_dir = Path(r"R:\sebastian\backend\quantum_framework\personality\training\data_collection\analysis")
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Save JSON data
        json_file = output_dir / f'character_dialogue_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
        # Generate readable report
        report_file = output_dir / f'dialogue_report_{timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== BLACK BUTLER SEASON 1 DIALOGUE ANALYSIS ===\n\n")
            for char, dialogues in results.items():
                f.write(f"\n{char.upper()} - Total lines: {len(dialogues)}\n")
                f.write("="* 50 + "\n")
                for dialogue in dialogues[:5]:  # Show sample of dialogues
                    f.write(f"Episode: {dialogue['episode']}\n")
                    f.write(f"Line: {dialogue['text']}\n")
                    f.write("-"* 30 + "\n")

if __name__ == "__main__":
    identifier = DialogueIdentifier()
    results = identifier.process_transcript_directory()
    identifier.save_dialogue_analysis(results)