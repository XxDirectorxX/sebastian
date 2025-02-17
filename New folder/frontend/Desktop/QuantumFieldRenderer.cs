using System;
using System.Windows.Media.Media3D;
using System.Windows.Media;

namespace Sebastian
{
    public class QuantumFieldRenderer
    {
        private readonly Viewport3D _viewport;
        private readonly Model3DGroup _quantumField;
        private readonly double[,,] _fieldState;
        private readonly double _fieldStrength = 46.97871376;
        
        public QuantumFieldRenderer(Viewport3D viewport)
        {
            _viewport = viewport;
            _quantumField = new Model3DGroup();
            _fieldState = new double[64, 64, 64];
            
            InitializeFieldGeometry();
        }

        private void InitializeFieldGeometry()
        {
            var geometry = new MeshGeometry3D();
            
            // Generate field visualization mesh
            for (int x = 0; x < 64; x++)
            for (int y = 0; y < 64; y++)
            for (int z = 0; z < 64; z++)
            {
                AddQuantumPoint(geometry, x, y, z);
            }
            
            var material = new DiffuseMaterial(new SolidColorBrush(Colors.Blue));
            var model = new GeometryModel3D(geometry, material);
            _quantumField.Children.Add(model);
        }

        public void RenderQuantumField(double[,,] quantumState)
        {
            Array.Copy(quantumState, _fieldState, quantumState.Length);
            UpdateFieldVisualization();
        }

        private void UpdateFieldVisualization()
        {
            var geometry = (_quantumField.Children[0] as GeometryModel3D).Geometry as MeshGeometry3D;
            
            for (int i = 0; i < geometry.Positions.Count; i++)
            {
                var pos = geometry.Positions[i];
                var fieldValue = _fieldState[(int)pos.X, (int)pos.Y, (int)pos.Z];
                var newPos = ApplyQuantumTransform(pos, fieldValue);
                geometry.Positions[i] = newPos;
            }
        }

        private Point3D ApplyQuantumTransform(Point3D point, double fieldValue)
        {
            var phase = Math.Exp(fieldValue * _fieldStrength);
            return new Point3D(
                point.X * phase,
                point.Y * phase,
                point.Z * phase
            );
        }

        public void UpdateFieldDynamics()
        {
            var rotation = new AxisAngleRotation3D(new Vector3D(0, 1, 0), 0.5);
            _quantumField.Transform = new RotateTransform3D(rotation);
        }
    }
}
