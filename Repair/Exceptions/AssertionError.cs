namespace GPURepair.Repair.Exceptions
{
    using System;

    public class AssertionError : Exception
    {
        public AssertionError()
            : base()
        {
        }

        public AssertionError(string message)
            : base(message)
        {
        }
    }
}
