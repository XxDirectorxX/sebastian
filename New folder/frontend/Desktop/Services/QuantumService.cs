using System;
using System.Threading.Tasks;
using System.Net.WebSockets;
using Newtonsoft.Json;

namespace Sebastian.Services
{
    public class QuantumService
    {
        private readonly WebSocketClient _wsClient;
        private readonly double _fieldStrength = 46.97871376;
        
        public QuantumService()
        {
            _wsClient = new WebSocketClient("ws://localhost:8000/ws/quantum");
        }

        public async Task InitializeQuantumConnection()
        {
            await _wsClient.ConnectAsync();
            await StartQuantumStream();
        }

        private async Task StartQuantumStream()
        {
            while (true)
            {
                var state = await _wsClient.ReceiveQuantumStateAsync();
                OnQuantumStateReceived?.Invoke(state);
            }
        }

        public event Action<QuantumState> OnQuantumStateReceived;
    }
}
