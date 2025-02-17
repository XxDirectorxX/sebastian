namespace Sebastian.Shared.Models
{
    public class FieldStrength
    {
        public const double BaseStrength = 46.97871376;
        public const double RealityCoherence = 1.618033988749895;
        public const int MatrixDimension = 64;
        
        public double CurrentStrength { get; private set; }
        
        public void AdjustStrength(double coherenceFactor)
        {
            CurrentStrength = BaseStrength * coherenceFactor;
        }
    }
}
