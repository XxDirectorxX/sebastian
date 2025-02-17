using System.Windows.Controls;
using System.Windows.Media.Media3D;

namespace Sebastian.Controls
{
    public partial class QuantumFieldControl : UserControl
    {
        private readonly QuantumFieldRenderer _renderer;
        private readonly double _fieldStrength = 46.97871376;
        
        public QuantumFieldControl()
        {
            InitializeComponent();
            _renderer = new QuantumFieldRenderer(QuantumViewport);
            InitializeQuantumField();
        }

        private void InitializeQuantumField()
        {
            var geometry = new MeshGeometry3D();
            for (int x = 0; x < 64; x++)
            for (int y = 0; y < 64; y++)
            for (int z = 0; z < 64; z++)
            {
                _renderer.AddQuantumPoint(geometry, x, y, z);
            }
        }

        public void UpdateQuantumState(double[,,] state)
        {
            _renderer.RenderQuantumField(state);
        }
    }
}