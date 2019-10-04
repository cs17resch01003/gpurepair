namespace GPURepair.Repair.Exceptions
{
    using System;

    public class RepairError : Exception
    {
        public RepairError()
            : base()
        {
        }

        public RepairError(string message)
            : base(message)
        {
        }
    }
}
