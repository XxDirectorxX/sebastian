using System.ComponentModel;
using System.Windows.Input;
using Sebastian.Models;
using Sebastian.Services;

namespace Sebastian.ViewModels
{
    public class MainViewModel : INotifyPropertyChanged
    {
        private readonly QuantumService _quantumService;
        private QuantumState _currentState;
        
        public QuantumState CurrentState
        {
            get => _currentState;
            set
            {
                _currentState = value;
                OnPropertyChanged(nameof(CurrentState));
            }
        }

        public MainViewModel()
        {
            _quantumService = new QuantumService();
            _quantumService.OnQuantumStateReceived += UpdateQuantumState;
            InitializeCommands();
        }

        private void UpdateQuantumState(QuantumState state)
        {
            CurrentState = state;
        }

        public event PropertyChangedEventHandler PropertyChanged;
        protected void OnPropertyChanged(string name)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }
}
