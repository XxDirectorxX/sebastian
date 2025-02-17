using System;
using System.Speech.Recognition;
using System.Speech.Synthesis;

namespace Sebastian
{
    public class VoiceProcessor
    {
        private readonly SpeechRecognitionEngine _recognizer;
        private readonly SpeechSynthesizer _synthesizer;
        
        public event Action<string> OnCommandReceived;
        
        public VoiceProcessor()
        {
            _recognizer = new SpeechRecognitionEngine();
            _synthesizer = new SpeechSynthesizer();
            
            InitializeVoiceSystem();
        }

        private void InitializeVoiceSystem()
        {
            var commands = new Choices();
            commands.Add(new string[] {
                "Initialize quantum field",
                "Stabilize reality",
                "Adjust field strength",
                "Monitor coherence",
                "Execute protocol"
            });
            
            var grammarBuilder = new GrammarBuilder(commands);
            var grammar = new Grammar(grammarBuilder);
            
            _recognizer.LoadGrammar(grammar);
            _recognizer.SpeechRecognized += HandleSpeechRecognized;
            
            _synthesizer.SelectVoiceByHints(VoiceGender.Male, VoiceAge.Adult);
            _synthesizer.Rate = 0;
            _synthesizer.Volume = 100;
        }

        public void Initialize()
        {
            _recognizer.SetInputToDefaultAudioDevice();
            _recognizer.RecognizeAsync(RecognizeMode.Multiple);
        }

        private void HandleSpeechRecognized(object sender, SpeechRecognizedEventArgs e)
        {
            if (e.Result.Confidence > 0.8)
            {
                OnCommandReceived?.Invoke(e.Result.Text);
                RespondToCommand(e.Result.Text);
            }
        }

        private void RespondToCommand(string command)
        {
            string response = command switch
            {
                "Initialize quantum field" => "Initializing quantum field, my lord",
                "Stabilize reality" => "Stabilizing reality parameters",
                "Adjust field strength" => "Adjusting field strength to optimal levels",
                "Monitor coherence" => "Monitoring reality coherence",
                "Execute protocol" => "Executing quantum protocol",
                _ => "Command acknowledged"
            };
            
            _synthesizer.SpeakAsync(response);
        }
    }
}
