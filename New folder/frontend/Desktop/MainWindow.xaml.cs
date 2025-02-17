using System;
using System.Windows;
using System.Windows.Media.Media3D;
using System.Windows.Threading;

namespace Sebastian
{
    public partial class MainWindow : Window
    {
        private readonly double FIELD_STRENGTH = 46.97871376;
        private readonly double REALITY_COHERENCE = 1.618033988749895;
        private readonly QuantumFieldRenderer _fieldRenderer;
        private readonly WebSocketClient _wsClient;
        private readonly VoiceProcessor _voiceProcessor;
        
        public MainWindow()
        {
            InitializeComponent();
            
            _fieldRenderer = new QuantumFieldRenderer(QuantumFieldViewport);
            _wsClient = new WebSocketClient("ws://localhost:8000/ws/quantum");
            _voiceProcessor = new VoiceProcessor();
            
            InitializeQuantumSystem();
            StartRealityMonitoring();
        }

        private async void InitializeQuantumSystem()
        {
            await _wsClient.ConnectAsync();
            _wsClient.OnQuantumStateReceived += UpdateQuantumVisualization;
            _wsClient.OnRealityMetricsReceived += UpdateRealityMetrics;
            
            _voiceProcessor.Initialize();
            _voiceProcessor.OnCommandReceived += ProcessVoiceCommand;
        }

        private void UpdateQuantumVisualization(double[,,] quantumState)
        {
            Dispatcher.InvokeAsync(() =>
            {
                _fieldRenderer.RenderQuantumField(quantumState);
                UpdateFieldMetrics(quantumState);
            });
        }

        private void UpdateRealityMetrics(double coherence, double fieldStrength)
        {
            Dispatcher.InvokeAsync(() =>
            {
                CoherenceBar.Value = coherence;
                FieldStrengthValue.Text = fieldStrength.ToString("F8");
            });
        }

        private void StartRealityMonitoring()
        {
            var timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(16.667) }; // 60 FPS
            timer.Tick += (s, e) => _fieldRenderer.UpdateFieldDynamics();
            timer.Start();
        }

        private async void ProcessVoiceCommand(string command)
        {
            CommandHistory.Items.Add($"{DateTime.Now:HH:mm:ss} - {command}");
            await _wsClient.SendCommandAsync(command);
        }
    }
}