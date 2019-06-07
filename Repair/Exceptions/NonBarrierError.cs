namespace GPURepair.Repair.Exceptions
{
    using System;

    public class NonBarrierError: Exception
    {
        public NonBarrierError()
            : base()
        {
        }

        public NonBarrierError(string message)
            : base(message)
        {
        }
    }
}
