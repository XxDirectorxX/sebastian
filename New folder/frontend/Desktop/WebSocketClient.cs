using System;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace Sebastian
{
    public class WebSocketClient
    {
        private readonly ClientWebSocket _ws;
        private readonly string _url;
        private readonly CancellationTokenSource _cts;
        
        public event Action<double[,,]> OnQuantumStateReceived;
        public event Action<double, double> OnRealityMetricsReceived;
        
        public WebSocketClient(string url)
        {
            _url = url;
            _ws = new ClientWebSocket();
            _cts = new CancellationTokenSource();
        }

        public async Task ConnectAsync()
        {
            await _ws.ConnectAsync(new Uri(_url), _cts.Token);
            _ = ReceiveLoop();
        }

        private async Task ReceiveLoop()
        {
            var buffer = new byte[64 * 1024];
            
            try
            {
                while (_ws.State == WebSocketState.Open)
                {
                    var result = await _ws.ReceiveAsync(buffer, _cts.Token);
                    if (result.MessageType == WebSocketMessageType.Text)
                    {
                        var json = Encoding.UTF8.GetString(buffer, 0, result.Count);
                        ProcessQuantumMessage(json);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"WebSocket error: {ex.Message}");
            }
        }

        private void ProcessQuantumMessage(string json)
        {
            var data = JsonConvert.DeserializeObject<QuantumMessage>(json);
            OnQuantumStateReceived?.Invoke(data.QuantumState);
            OnRealityMetricsReceived?.Invoke(data.Coherence, data.FieldStrength);
        }

        public async Task SendCommandAsync(string command)
        {
            var json = JsonConvert.SerializeObject(new { command });
            var buffer = Encoding.UTF8.GetBytes(json);
            await _ws.SendAsync(buffer, WebSocketMessageType.Text, true, _cts.Token);
        }

        private class QuantumMessage
        {
            public double[,,] QuantumState { get; set; }
            public double Coherence { get; set; }
            public double FieldStrength { get; set; }
        }
    }
}
