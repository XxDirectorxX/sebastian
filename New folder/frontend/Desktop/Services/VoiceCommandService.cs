using System.Speech.Recognition;
using System.Speech.Synthesis;

namespace Sebastian.Services
{
    public class VoiceCommandService
    {
        private readonly SpeechRecognitionEngine _recognizer;
        private readonly SpeechSynthesizer _synthesizer;
        
        public VoiceCommandService()
        {
            _recognizer = new SpeechRecognitionEngine();
            _synthesizer = new SpeechSynthesizer();
            InitializeVoiceSystem();
        }

        private void InitializeVoiceSystem()
        {
            var commands = new Choices(new string[] {
                "Initialize quantum field",
                "Adjust field strength",
                "Monitor coherence",
                "Execute protocol"
            });
            
            var grammarBuilder = new GrammarBuilder(commands);
            _recognizer.LoadGrammar(new Grammar(grammarBuilder));
            _recognizer.SetInputToDefaultAudioDevice();
            
            _synthesizer.SelectVoiceByHints(VoiceGender.Male);
            _synthesizer.Rate = 0;
            _synthesizer.Volume = 100;
        }

        public void StartListening()
        {
            _recognizer.RecognizeAsync(RecognizeMode.Multiple);
        }
    }
}
