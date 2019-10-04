namespace GPURepair.Repair.Exceptions
{
    using System;

    public class SummaryGeneratorError : Exception
    {
        public SummaryGeneratorError()
            : base()
        {
        }

        public SummaryGeneratorError(string message)
            : base(message)
        {
        }
    }
}
