namespace GPURepair.Repair.Exceptions
{
    using System;

    public class InconclusiveError : Exception
    {
        public InconclusiveError()
            : base()
        {
        }

        public InconclusiveError(string message)
            : base(message)
        {
        }
    }
}
