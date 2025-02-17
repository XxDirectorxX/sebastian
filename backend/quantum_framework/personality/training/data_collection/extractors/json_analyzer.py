from pathlib import Path
import json
import time

class JsonAnalyzer:
    def __init__(self):
        self.context_dir = Path(r"R:\sebastian\backend\quantum_framework\personality\training\data_collection\raw_data\context")
        self.results_dir = Path(r"R:\sebastian\backend\quantum_framework\personality\training\data_collection\analysis")
        self.results_dir.mkdir(exist_ok=True)

    def analyze_json_files(self):
        analysis = {
            'dialogue_patterns': {
                'formal_responses': [],
                'combat_dialogue': [],
                'master_interactions': [],
                'demon_nature': []
            },
            'personality_traits': {
                'butler_aspects': [],
                'demonic_aspects': [],
                'intellectual_traits': []
            },
            'behavioral_triggers': {
                'duty_responses': [],
                'pride_responses': [],
                'threat_responses': []
            },
            'relationship_dynamics': {
                'with_ciel': [],
                'with_servants': [],
                'with_enemies': []
            }
        }

        for json_file in self.context_dir.glob('*.json'):
            self.process_file(json_file, analysis)

        self.save_refined_analysis(analysis)
        return analysis

    def process_file(self, file_path, analysis):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.categorize_content(data, analysis)

    def categorize_content(self, data, analysis):
        # Add categorization logic here
        pass

    def save_refined_analysis(self, analysis):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Save detailed analysis
        output_file = self.results_dir / f'refined_analysis_{timestamp}.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== SEBASTIAN MICHAELIS CHARACTER ANALYSIS ===\n\n")
            
            for category, subcategories in analysis.items():
                f.write(f"\n{category.upper()}\n")
                f.write("=" * len(category) + "\n")
                
                for subcat, items in subcategories.items():
                    f.write(f"\n{subcat}:\n")
                    f.write("-" * len(subcat) + "\n")
                    for item in items:
                        f.write(f"• {item}\n")
                    f.write("\n")

if __name__ == "__main__":
    analyzer = JsonAnalyzer()
    analyzer.analyze_json_files()